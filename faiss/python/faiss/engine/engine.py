# generator initialization built upon FlashRAG
from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict, deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import threading
import uuid
from pathlib import Path
import faiss
import time
import numpy as np
from heapq import heappush, heappop

from ..utils import *
from .encoder import Encoder, STEncoder
from .scheduler import BaseScheduler, SchedulerRequest, FifoScheduler, RoundRobinScheduler


class EngineError(RuntimeError):
    pass


class AsyncNotEnabledError(EngineError):
    pass


class AsyncAlreadyEnabledError(EngineError):
    pass


class AsyncShutdownError(EngineError):
    pass


class EvictionPolicyError(EngineError):
    pass


class _EvictionTracker:
    def __init__(self, policy: str):
        if policy is None:
            raise EvictionPolicyError("eviction policy must be provided")
        policy_norm = str(policy).strip().lower()
        if policy_norm not in {"lru", "fifo", "lfu"}:
            raise EvictionPolicyError(
                f"unsupported eviction policy: {policy}. "
                "Use 'lru', 'fifo', or 'lfu'"
            )
        self._policy = policy_norm
        self._lru = OrderedDict() if policy_norm == "lru" else None
        self._fifo_queue = deque() if policy_norm == "fifo" else None
        self._fifo_set = set() if policy_norm == "fifo" else None
        self._freq = {} if policy_norm == "lfu" else None

    def record_access(self, list_ids: List[int]) -> None:
        for list_id in list_ids:
            self.record_loaded(list_id)

    def record_loaded(self, list_id: int) -> None:
        if self._policy == "lru":
            self._lru.pop(list_id, None)
            self._lru[list_id] = None
        elif self._policy == "fifo":
            if list_id in self._fifo_set:
                return
            self._fifo_queue.append(list_id)
            self._fifo_set.add(list_id)
        else:  # lfu
            current = self._freq.get(list_id, 0)
            self._freq[list_id] = current + 1

    def remove(self, list_id: int) -> None:
        if self._policy == "lru":
            self._lru.pop(list_id, None)
        elif self._policy == "fifo":
            if list_id in self._fifo_set:
                self._fifo_set.remove(list_id)
        else:  # lfu
            self._freq.pop(list_id, None)

    def pop_victim(self, protected: set) -> Optional[int]:
        if self._policy == "lru":
            for list_id in list(self._lru.keys()):
                if list_id in protected:
                    continue
                self._lru.pop(list_id, None)
                return list_id
            return None

        if self._policy == "fifo":
            while self._fifo_queue:
                list_id = self._fifo_queue.popleft()
                if list_id not in self._fifo_set:
                    continue
                self._fifo_set.remove(list_id)
                if list_id in protected:
                    continue
                return list_id
            return None

        # LFU: select the least frequently used list that is not protected.
        if not self._freq:
            return None
        candidate = None
        candidate_freq = None
        for lid, freq in self._freq.items():
            if lid in protected:
                continue
            if candidate is None or freq < candidate_freq or (
                freq == candidate_freq and lid < candidate
            ):
                candidate = lid
                candidate_freq = freq
        if candidate is None:
            return None
        self._freq.pop(candidate, None)
        return candidate


class _BackendLfuPolicy:
    """Backend-driven LFU-style eviction based on probe_count."""

    def select_eviction_candidate(
        self, stats: Dict[int, Dict[str, int]], protected: set
    ) -> Optional[int]:
        candidate: Optional[int] = None
        candidate_freq: Optional[int] = None
        for lid, s in stats.items():
            if lid in protected:
                continue
            freq = int(s.get("probe_count", 0))
            if candidate is None or freq < candidate_freq or (
                freq == candidate_freq and lid < candidate
            ):
                candidate = lid
                candidate_freq = freq
        return candidate


class _BackendLruPolicy:
    """Backend-driven LRU-style eviction based on last_probe_ts."""

    def select_eviction_candidate(
        self, stats: Dict[int, Dict[str, int]], protected: set
    ) -> Optional[int]:
        candidate: Optional[int] = None
        candidate_ts: Optional[int] = None
        for lid, s in stats.items():
            if lid in protected:
                continue
            ts = int(s.get("last_probe_ts", 0))
            if candidate is None or ts < candidate_ts or (
                ts == candidate_ts and lid < candidate
            ):
                candidate = lid
                candidate_ts = ts
        return candidate


@dataclass
class FaissEnginConfig(BaseConfig):
    """Config for all engines."""
    index_type: str
    index_path: str
    corpus_path: str

    retrieval_method: str
    retrieval_topk: int
    retrieval_batch_size: int
    retrieval_query_max_length: int
    retrieval_use_fp16: bool
    retrieval_pooling_method: str

    return_embedding: bool
    larger_topk: int

    use_sentence_transformer: bool

    # GPU Config
    gpu_memory_utilization: float
    eviction_policy: str
    eviction_max_attempts: int

    # IVF Config
    nprobe: int

    # Optional (must follow all required fields for dataclass)
    retrieval_model_path: Optional[str] = None

    # Async / scheduler config (optional, kept backward compatible)
    async_enabled: bool = False
    async_scheduler_type: str = "fifo"  # "fifo" | "round_robin" | "priority"
    async_scheduler_params: Dict[str, Any] = field(default_factory=dict)
    async_max_batch_size: Optional[int] = None
    async_max_wait_ms: int = 0
    async_step_interval_ms: int = 0
    async_queue_size: int = 1024
    async_idle_sleep_s: float = 0.01

    # Eviction tuning (optional)
    # GPU memory / miss-policy tuning (optional)
    # gpu_watermark_high controls the maximum fraction of IVF lists that we
    # allow to reside on GPU before proactively evicting to preserve headroom
    # in the GPU memory pool for auto-fetch traffic.
    gpu_memory_policy: str = "none"  # "none" | "static_topk" | "dynamic_watermark"
    gpu_memory_target_fraction: float = 0.0
    gpu_watermark_high: float = 0.0
    gpu_miss_policy: str = "error"  # "error" | "auto_fetch" | "cpu_offload"
    gpu_no_copy_evict: bool = False

    # Multi-GPU build (optional)
    # When enabled, Engine will move the CPU index to GPU(s) at runtime.
    # The default remains CPU-only to preserve backward compatibility.
    gpu_enabled: bool = False
    gpu_ngpu: int = 0
    gpu_devices: Optional[List[int]] = None
    gpu_shard: bool = True
    gpu_shard_type: int = 5
    gpu_common_ivf_quantizer: bool = True
    gpu_use_cuvs: bool = False

    # GPU IVF initialization (optional)
    # Controls which IVF clusters are proactively loaded to GPU at startup.
    # Supported strategies:
    #   "none"      : do not preload any IVF lists.
    #   "random"    : randomly sample IVF lists until the memory pool is full.
    #   "largest"   : load largest IVF lists first (by cluster size).
    #   "frequency" : load IVF lists ordered by historical access frequency
    #                 (see engine/config/default.json).
    gpu_ivf_init_strategy: str = "none"

    # Backend-driven IVF cluster stats / policy (optional)
    cluster_stats_refresh_interval_ms: int = 100
    cluster_policy_type: str = "backend_lfu"  # "backend_lfu" | "backend_lru" | "none"


class BaseEngine(ABC):
    """Base engine for all retrievers."""

    def __init__(self, config: BaseConfig):
        self.config = config
        self._search_lock = threading.Lock()
        self._async_controller = None
        
    @abstractmethod
    def _search(self, query: str, num: int, return_score: bool) -> List[Dict[str, str]]:
        r"""Retrieve topk relevant documents in corpus.

        Return:
            list: contains information related to the document, including:
                contents: used for building index
                title: (if provided)
                text: (if provided)
        """
        pass

    @abstractmethod
    def _batch_search(self, query_list, query_id_list, num, return_score, eval_cache):
        pass

    @abstractmethod
    def _add(self, query: str, return_centroid: bool, retrain: bool):
        pass

    @abstractmethod
    def _batch_add(self, query_list, return_centroid: bool, retrain: bool):
        pass

    def search(self, query: str, num: Optional[int] = None, return_score: bool = False):
        if not isinstance(query, str) or not query:
            raise ValueError("query must be a non-empty string")
        with self._search_lock:
            return self._search(query, num=num, return_score=return_score)

    def batch_search(
        self,
        query_list: List[str],
        num: Optional[int] = None,
        return_score: bool = False,
        eval_cache: bool = False,
    ):
        if query_list is None or len(query_list) == 0:
            raise ValueError("query_list must not be empty")
        with self._search_lock:
            return self._batch_search(
                query_list=query_list,
                query_id_list=None,
                num=num,
                return_score=return_score,
                eval_cache=eval_cache,
            )

    def enable_async(
        self,
        scheduler: BaseScheduler,
        max_queue_size: int = 1024,
        idle_sleep_s: float = 0.01,
    ) -> None:
        if self._async_controller is not None:
            raise AsyncAlreadyEnabledError("async controller already enabled")
        self._async_controller = _AsyncBatchingController(
            engine=self,
            scheduler=scheduler,
            max_queue_size=max_queue_size,
            idle_sleep_s=idle_sleep_s,
        )
        self._async_controller.start()

    def shutdown_async(self, drain: bool = True) -> None:
        if self._async_controller is None:
            return
        self._async_controller.shutdown(drain=drain)
        self._async_controller = None

    def is_async_enabled(self) -> bool:
        return self._async_controller is not None

    def search_async(
        self,
        query: str,
        num: Optional[int] = None,
        return_score: bool = False,
        source_id: str = "default",
    ) -> Future:
        if self._async_controller is None:
            raise AsyncNotEnabledError("async controller not enabled")
        return self._async_controller.submit(
            query=query,
            num=num,
            return_score=return_score,
            source_id=source_id,
        )

    def batch_search_async(
        self,
        query_list: List[str],
        num: Optional[int] = None,
        return_score: bool = False,
        source_id: str = "default",
    ) -> List[Future]:
        if query_list is None or len(query_list) == 0:
            raise ValueError("query_list must not be empty")
        futures = []
        for query in query_list:
            futures.append(
                self.search_async(
                    query=query,
                    num=num,
                    return_score=return_score,
                    source_id=source_id,
                )
            )
        return futures


class _AsyncBatchingController:
    def __init__(
        self,
        engine: BaseEngine,
        scheduler: BaseScheduler,
        max_queue_size: int,
        idle_sleep_s: float,
    ):
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be > 0")
        if idle_sleep_s <= 0:
            raise ValueError("idle_sleep_s must be > 0")
        self._engine = engine
        self._scheduler = scheduler
        self._max_queue_size = max_queue_size
        self._idle_sleep_s = idle_sleep_s
        self._shutdown_event = threading.Event()
        self._wake_event = threading.Event()
        self._worker = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._worker.start()

    def shutdown(self, drain: bool = True) -> None:
        self._shutdown_event.set()
        self._wake_event.set()
        self._worker.join()
        if drain:
            remaining = self._scheduler.flush()
            if remaining:
                self._process_batch(remaining)

    def submit(
        self,
        query: str,
        num: Optional[int],
        return_score: bool,
        source_id: str,
    ) -> Future:
        if self._shutdown_event.is_set():
            raise AsyncShutdownError("async controller is shut down")
        if not isinstance(query, str) or not query:
            raise ValueError("query must be a non-empty string")
        if source_id is None or source_id == "":
            raise ValueError("source_id must be non-empty")

        if self._scheduler.pending_count() >= self._max_queue_size:
            raise EngineError("async queue is full")

        request = SchedulerRequest(
            id=str(uuid.uuid4()),
            query=query,
            num=num,
            return_score=return_score,
            enqueue_time=time.monotonic(),
            source_id=source_id,
            future=Future(),
        )
        self._scheduler.add_request(request)
        self._wake_event.set()
        return request.future

    def _loop(self) -> None:
        while not self._shutdown_event.is_set():
            now = time.monotonic()
            if self._scheduler.should_step(now):
                batch = self._scheduler.step(now)
                if batch:
                    self._process_batch(batch)
                    continue
            self._wake_event.wait(timeout=self._idle_sleep_s)
            self._wake_event.clear()

    def _process_batch(self, batch: List[SchedulerRequest]) -> None:
        if not batch:
            return
        grouped: Dict[tuple, List[SchedulerRequest]] = defaultdict(list)
        for request in batch:
            grouped[(request.num, request.return_score)].append(request)

        for (num, return_score), requests in grouped.items():
            queries = [req.query for req in requests]
            try:
                with self._engine._search_lock:
                    result = self._engine._batch_search(
                        query_list=queries,
                        query_id_list=requests,
                        num=num,
                        return_score=return_score,
                        eval_cache=False,
                    )
                if return_score:
                    results, scores = result
                    if len(results) != len(requests) or len(scores) != len(requests):
                        raise EngineError("batch search returned mismatched result sizes")
                    for req, res, score in zip(requests, results, scores):
                        if not req.future.done():
                            req.future.set_result((res, score))
                else:
                    results = result
                    if len(results) != len(requests):
                        raise EngineError("batch search returned mismatched result sizes")
                    for req, res in zip(requests, results):
                        if not req.future.done():
                            req.future.set_result(res)
            except Exception as exc:
                for req in requests:
                    if not req.future.done():
                        req.future.set_exception(exc)


class FaissEngine(BaseEngine):
    r"""Dense retriever based on pre-built faiss index."""

    def __init__(self, config: FaissEnginConfig):
        super().__init__(config)
        if config.retrieval_model_path is None:
            raise EngineError("retrieval_model_path must be provided in engine_config")
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk
        # TODO: Handle different retrieval top-k

        self.batch_size = config.retrieval_batch_size
        self.return_embedding = config.return_embedding
        self.index_path = config.index_path
        self.corpus_path = config.corpus_path

        self.index = faiss.read_index(self.index_path)
        # Keep a reference to the original CPU index when we later build a GPU
        # view, as CpuOffload uses external CPU backing.
        self._cpu_index = self.index
        self._gpu_resources = None

        self.corpus = load_corpus(self.corpus_path)

        if config.use_sentence_transformer:
            self.encoder = STEncoder(
                model_name=self.retrieval_method,
                model_path=config.retrieval_model_path,
                max_length=config.retrieval_query_max_length,
                use_fp16=config.retrieval_use_fp16,
            )
        else:
            self.encoder = Encoder(
                model_name=self.retrieval_method,
                model_path=config.retrieval_model_path,
                pooling_method=config.retrieval_pooling_method,
                max_length=config.retrieval_query_max_length,
                use_fp16=config.retrieval_use_fp16,
            )
        
        # Optionally move index to GPU(s) before initializing policies.
        self._maybe_build_gpu_index(config)
        self.reinit(config)

        if getattr(config, "async_enabled", False):
            scheduler = getattr(config, "async_scheduler", None)
            if scheduler is None:
                # Select scheduler implementation based on config.async_scheduler_type
                max_batch_size = (
                    config.async_max_batch_size
                    if config.async_max_batch_size is not None
                    else self.batch_size
                )
                max_wait_ms = getattr(config, "async_max_wait_ms", 0)
                step_interval_ms = getattr(config, "async_step_interval_ms", 0)
                scheduler_type = str(
                    getattr(config, "async_scheduler_type", "fifo")
                ).strip().lower()

                if scheduler_type == "fifo":
                    scheduler = FifoScheduler(
                        max_batch_size=max_batch_size,
                        max_wait_ms=max_wait_ms,
                        step_interval_ms=step_interval_ms,
                    )
                elif scheduler_type == "round_robin":
                    scheduler = RoundRobinScheduler(
                        max_batch_size=max_batch_size,
                        max_wait_ms=max_wait_ms,
                        step_interval_ms=step_interval_ms,
                    )
                elif scheduler_type == "priority":
                    from .scheduler import PriorityScheduler  # local import to avoid cycle

                    params = getattr(config, "async_scheduler_params", {}) or {}
                    source_priorities = params.get("source_priorities")
                    default_priority = params.get("default_priority", 0)
                    scheduler = PriorityScheduler(
                        max_batch_size=max_batch_size,
                        max_wait_ms=max_wait_ms,
                        step_interval_ms=step_interval_ms,
                        source_priorities=source_priorities,
                        default_priority=default_priority,
                    )
                else:
                    raise EngineError(
                        f"unsupported async_scheduler_type: {scheduler_type!r}"
                    )

            if not isinstance(scheduler, BaseScheduler):
                raise EngineError("async_scheduler must be a BaseScheduler instance")
            self.enable_async(
                scheduler=scheduler,
                max_queue_size=getattr(config, "async_queue_size", 1024),
                idle_sleep_s=getattr(config, "async_idle_sleep_s", 0.01),
            )

    def _maybe_build_gpu_index(self, config: FaissEnginConfig) -> None:
        """Optionally move the loaded CPU index to GPU(s)."""
        if not bool(getattr(config, "gpu_enabled", False)):
            return

        if not hasattr(faiss, "get_num_gpus") or faiss.get_num_gpus() <= 0:
            raise EngineError("gpu_enabled=True but no GPUs are available")

        # Avoid rebuilding if the index already appears to be GPU-backed.
        # (We keep this heuristic conservative.)
        if self.index is None:
            raise EngineError("index is not initialized")

        cpu_index = self.index

        devices = getattr(config, "gpu_devices", None)
        ngpu = int(getattr(config, "gpu_ngpu", 0) or 0)
        if devices is None:
            if ngpu <= 0:
                # Backward-compatible default: use GPU 0.
                devices = [0]
            else:
                devices = list(range(ngpu))
        else:
            devices = list(devices)

        if not devices:
            raise EngineError("gpu_devices must not be empty when gpu_enabled=True")

        # Validate device IDs
        for d in devices:
            if int(d) < 0:
                raise EngineError(f"invalid GPU device id: {d!r}")

        # Normalize miss policy early as it affects how we build sharded indices.
        miss_policy = str(getattr(config, "gpu_miss_policy", "error") or "").strip().lower()

        # Single GPU
        if len(devices) == 1:
            dev = int(devices[0])
            res = faiss.StandardGpuResources()
            try:
                self.index = faiss.index_cpu_to_gpu(res, dev, cpu_index)
            except Exception as exc:
                raise EngineError(f"failed to move index to GPU {dev}") from exc
            self._gpu_resources = [res]
            return

        # Multi GPU
        if not bool(getattr(config, "gpu_shard", True)):
            raise EngineError(
                "multi-GPU replicas are not implemented in Engine; "
                "set gpu_shard=True"
            )

        shard_type = int(getattr(config, "gpu_shard_type", 5) or 5)
        common_ivf_quantizer = bool(getattr(config, "gpu_common_ivf_quantizer", True))
        use_cuvs = bool(getattr(config, "gpu_use_cuvs", False))

        res_list = [faiss.StandardGpuResources() for _ in devices]
        self._gpu_resources = res_list

        if miss_policy in {"cpu_offload", "cpuoffload"}:
            if not hasattr(faiss, "build_sharded_ivfflat_cpu_offload_index"):
                raise EngineError(
                    "gpu_miss_policy='cpu_offload' requires "
                    "build_sharded_ivfflat_cpu_offload_index helper"
                )
            try:
                index_shards, cpu_backings, _lis, _l2s = (
                    faiss.build_sharded_ivfflat_cpu_offload_index(
                        cpu_index,
                        gpus=devices,
                        resources=res_list,
                        shard_type=shard_type,
                        preload_list_ids_per_shard=None,
                        use_cuvs=use_cuvs,
                    )
                )
            except Exception as exc:
                raise EngineError("failed to build CpuOffload sharded GPU index") from exc

            # Keep explicit references as well (in addition to referenced_objects)
            # to avoid accidental GC in long-lived servers.
            self._cpu_shard_backings = cpu_backings
            self.index = index_shards
            return

        # Error / AutoFetch: can use the standard multi-GPU cloner
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.shard_type = shard_type
        co.common_ivf_quantizer = common_ivf_quantizer
        co.use_cuvs = use_cuvs
        try:
            self.index = faiss.index_cpu_to_gpu_multiple_py(res_list, cpu_index, co, devices)
        except Exception as exc:
            raise EngineError("failed to build sharded GPU index") from exc
    
    def reinit(self, config):
        self.config = config
        self.topk = config.retrieval_topk
        self.batch_size = self.config.retrieval_batch_size

        print(f"Index type {self.index.__class__.__name__}")

        # if (isinstance(self.index, faiss.swigfaiss_avx2.IndexIVFFlat)):
        if self.config.nprobe is not None:
            self.index.nprobe = self.config.nprobe
        else:
            self.index.nprobe = 16
        
        self.idxs_cache = dict()

        self.request_dict = {}

        # count when new centroid dist increases, the closest distance also increases.
        self.total_iter_clusters = 0
        self.total_overlap_clusters = 0
        self.total_far_clusters = 0
        self.total_far_dist = 0

        # the largest cluster which the topk document is in
        self.largest_cluster_ranking = []
        self.overlapped_rate = []
        self.overlapped_rate_top20 = []

        # document similarity search
        self.answer_in_doc_search = 0
        self.answer_in_doc_search_list = []
        self.all_answer_in_doc = 0
        self.all_answer_in_doc_list = []

        # old query similarity search
        self.answer_in_old_query_search = 0
        self.answer_in_old_query_search_list = []
        self.answer_not_in_old_query_search_list = []

        # answer in both search
        self.answer_in_both = 0

        # old query total search
        self.total_repeat_search = 0

        # time profile
        self.latency = []
        self.finished_requests = 0
        self.request_batch_size = 0
        self.in_old_centroid_but_not_in_local_buffer = 0
        self.in_old_centroid_and_in_local_buffer = 0
        self.in_old_centroid_top20_but_not_in_local_buffer = 0
        self.in_old_centroid_top20_and_in_local_buffer = 0

        self.in_old_hit_centroid_top20 = 0
        self.in_old_centroid_set = 0
        self.in_old_hit_centroid_top20_list = []
        self.in_old_centroid_set_list = []

        # test multi-request skewness
        self.skewness_dict = {i: 0 for i in range(self.index.nlist)}

        # profile document dist
        self.top1 = []
        self.top5 = []
        self.top20 = []

        # profile early termination
        self.old_termination_point = []
        self.new_termination_point = []
        self.new_termination_point2 = []
        self.new_termination_point3 = []

        self.larger_topk = config.larger_topk if config.larger_topk is not None else 20
        print(f"[Reordering setup]: topk as {self.topk} larger_topk as {self.larger_topk}")

        self._init_eviction_policy(config)
        # Configure GPU-side behavior (miss policy, memory hints)
        self._configure_gpu_miss_policy(config)
        self._configure_gpu_memory_policy(config)

        # Backend-driven cluster stats / policy state.
        self._cluster_stats_last_snapshot: Dict[int, Dict[str, int]] = {}
        self._cluster_stats_last_update_ts: float = 0.0
        self._cluster_policy = self._build_cluster_policy(config)

        # Optionally preload IVF lists to GPU based on configured strategy.
        self._initialize_gpu_ivf_lists(config)

    def _init_eviction_policy(self, config) -> None:
        policy = getattr(config, "eviction_policy", None)
        if policy is None or str(policy).strip().lower() in {"", "none", "disabled"}:
            self._eviction_enabled = False
            self._eviction_tracker = None
            self._eviction_max_attempts = 0
            return
        if not self._supports_gpu_eviction():
            raise EvictionPolicyError(
                "eviction policy requires a GPU IVF index with "
                "evict/load/list APIs available"
            )

        self._eviction_enabled = True
        self._eviction_tracker = _EvictionTracker(policy)
        self._eviction_max_attempts = int(
            getattr(config, "eviction_max_attempts", 64)
        )
        if self._eviction_max_attempts <= 0:
            raise EvictionPolicyError("eviction_max_attempts must be > 0")

        if hasattr(faiss, "set_auto_fetch"):
            try:
                # Disable auto-fetch on the concrete GPU IVF shards (single GPU
                # or IndexShardsIVF multi-GPU).
                any_shard = False
                for shard in self._iter_gpu_ivf_shards():
                    any_shard = True
                    faiss.set_auto_fetch(shard, False)
                if not any_shard:
                    faiss.set_auto_fetch(self.index, False)
            except Exception as exc:
                raise EvictionPolicyError(
                    "failed to disable auto-fetch before eviction control"
                ) from exc

    def _configure_gpu_miss_policy(self, config) -> None:
        """Configure GPU IVF list miss policy if supported by the index."""
        policy = getattr(config, "gpu_miss_policy", None)
        if policy is None:
            return

        policy_norm = str(policy).strip().lower()
        if policy_norm == "":
            return

        # When Python-side eviction is enabled, we intentionally disallow
        # GPU auto-fetch to avoid conflicting behaviors.
        if getattr(self, "_eviction_enabled", False) and policy_norm in {
            "auto_fetch",
            "autofetch",
        }:
            raise EvictionPolicyError(
                "gpu_miss_policy='auto_fetch' cannot be combined with a "
                "Python-side eviction_policy; disable eviction_policy or "
                "set gpu_miss_policy to 'error' or 'cpu_offload'"
            )

        # setMissPolicy exists on GpuIndexIVFFlat, but not on IndexShardsIVF.
        # For sharded multi-GPU, propagate the policy to each shard.
        has_direct = hasattr(self.index, "setMissPolicy")
        shards = list(self._iter_gpu_ivf_shards())
        if not has_direct and not shards:
            raise EngineError(
                "gpu_miss_policy requires a GPU IVF index exposing setMissPolicy "
                "(directly or via shards)"
            )

        mapping = {
            "error": 0,
            "auto_fetch": 1,
            "autofetch": 1,
            "cpu_offload": 2,
            "cpuoffload": 2,
        }
        if policy_norm not in mapping:
            raise EngineError(f"unsupported gpu_miss_policy: {policy!r}")

        try:
            # Enum is represented as an integer in Python bindings.
            if has_direct:
                self.index.setMissPolicy(mapping[policy_norm])
            else:
                for shard in shards:
                    shard.setMissPolicy(mapping[policy_norm])
        except Exception as exc:
            raise EngineError("failed to set GPU IVF miss policy") from exc

        # Optional no-copy eviction toggle.
        use_no_copy = bool(getattr(config, "gpu_no_copy_evict", False))
        if use_no_copy and hasattr(faiss, "set_no_copy_evict"):
            try:
                if shards:
                    for shard in shards:
                        faiss.set_no_copy_evict(shard, True)
                else:
                    faiss.set_no_copy_evict(self.index, True)
            except Exception as exc:
                raise EngineError(
                    "failed to enable GPU no-copy eviction"
                ) from exc

    def _configure_gpu_memory_policy(self, config) -> None:
        """Configure high-level GPU memory policy hints.

        Currently we implement a lightweight dynamic watermark policy based on the
        fraction of IVF lists resident on GPU. This avoids needing exact byte
        accounting while still providing a coarse control knob. The high watermark
        is interpreted as the maximum allowed fraction of IVF lists on GPU, so
        that the remaining capacity in the GPU memory pool can be reserved for
        auto-fetch traffic and transient loads.
        """
        policy = getattr(config, "gpu_memory_policy", "none")
        policy_norm = str(policy).strip().lower()
        self._gpu_memory_policy = policy_norm

        self._gpu_watermark_high = float(getattr(config, "gpu_watermark_high", 0.0))

        if policy_norm == "dynamic_watermark":
            if self._gpu_watermark_high <= 0.0 or self._gpu_watermark_high > 1.0:
                raise EngineError(
                    "gpu_memory_policy='dynamic_watermark' requires "
                    "0.0 < gpu_watermark_high <= 1.0"
                )
        elif policy_norm in {"none", "", "disabled"}:
            self._gpu_memory_policy = "none"
        elif policy_norm == "static_topk":
            # The current engine does not yet implement a full static_topk
            # preloading scheme at initialization time. We accept the setting
            # but do not change behavior here; callers can still use
            # simulate_onload_cluster for offline experiments.
            self._gpu_memory_policy = "static_topk"
        else:
            raise EngineError(f"unsupported gpu_memory_policy: {policy!r}")

    def _current_gpu_list_fraction(self) -> float:
        """Approximate fraction of IVF lists currently resident on GPU."""
        if not hasattr(self.index, "nlist") or not hasattr(faiss, "get_evicted_lists"):
            return 0.0
        total = int(getattr(self.index, "nlist", 0))
        if total <= 0:
            return 0.0
        try:
            evicted = faiss.get_evicted_lists(self.index)
        except Exception:
            return 0.0
        active = max(total - len(evicted), 0)
        return active / float(total)

    def _iter_gpu_ivf_shards(self):
        """
        Yield concrete GPU IVF shard indices if self.index is a multi-index
        container (e.g., IndexShardsIVF). Returns an empty iterator otherwise.
        """
        idx = getattr(self, "index", None)
        if idx is None:
            return iter(())
        if not (hasattr(idx, "count") and hasattr(idx, "at")):
            return iter(())

        try:
            n = int(idx.count())
        except Exception:
            return iter(())

        def _gen():
            for i in range(n):
                try:
                    shard = faiss.downcast_index(idx.at(i))
                except Exception:
                    continue
                # We only care about shards that look like GPU IVF list indices.
                if hasattr(shard, "setMissPolicy") or hasattr(shard, "evictCentroidToCpu"):
                    yield shard

        return _gen()

    def _supports_gpu_eviction(self) -> bool:
        return (
            hasattr(self.index, "isListOnGpu")
            and hasattr(self.index, "loadCentroidToGpu")
            and hasattr(self.index, "evictCentroidToCpu")
            and hasattr(self.index, "quantizer")
            and hasattr(faiss, "load_ivf_lists")
            and hasattr(faiss, "evict_ivf_lists")
            and hasattr(faiss, "is_list_on_gpu")
        )

    def _build_cluster_policy(self, config) -> Optional[object]:
        """Instantiate a backend-driven cluster policy if configured."""
        policy_type = str(
            getattr(config, "cluster_policy_type", "backend_lfu") or ""
        ).strip().lower()
        if policy_type in {"", "none", "disabled"}:
            return None
        if policy_type == "backend_lfu":
            return _BackendLfuPolicy()
        if policy_type == "backend_lru":
            return _BackendLruPolicy()
        raise EvictionPolicyError(
            f"unsupported cluster_policy_type: {policy_type!r}"
        )

    def _refresh_cluster_stats(self, force: bool = False) -> None:
        """Refresh cached IVF cluster stats from the backend if available."""
        if not hasattr(self, "index") or not hasattr(self.index, "nlist"):
            self._cluster_stats_last_snapshot = {}
            return
        if not hasattr(faiss, "get_list_activation_stats"):
            self._cluster_stats_last_snapshot = {}
            return

        interval_ms = int(
            getattr(self.config, "cluster_stats_refresh_interval_ms", 100) or 0
        )
        now_ms = time.monotonic() * 1000.0
        if (
            not force
            and interval_ms > 0
            and self._cluster_stats_last_update_ts > 0.0
            and now_ms - self._cluster_stats_last_update_ts < interval_ms
        ):
            return

        try:
            raw_stats = faiss.get_list_activation_stats(self.index)
        except Exception:
            # Be defensive: if backend stats are unavailable, clear snapshot
            self._cluster_stats_last_snapshot = {}
            self._cluster_stats_last_update_ts = now_ms
            return

        parsed: Dict[int, Dict[str, int]] = {}
        for entry in raw_stats or []:
            try:
                lid = int(entry.get("list_id"))  # type: ignore[call-arg]
            except Exception:
                continue
            parsed[lid] = {
                "probe_count": int(entry.get("probe_count", 0)),
                "load_count": int(entry.get("load_count", 0)),
                "last_probe_ts": int(entry.get("last_probe_ts", 0)),
            }

        self._cluster_stats_last_snapshot = parsed
        self._cluster_stats_last_update_ts = now_ms

    def get_cluster_stats(self) -> Dict[int, Dict[str, int]]:
        """Return the latest view of IVF cluster stats (backend-provided)."""
        self._refresh_cluster_stats(force=False)
        # Return a shallow copy to avoid accidental external mutation.
        return dict(self._cluster_stats_last_snapshot)

    def _ensure_lists_on_gpu(self, required_list_ids: List[int]) -> None:
        if not required_list_ids:
            return
        if not hasattr(faiss, "get_evicted_lists"):
            raise EvictionPolicyError("faiss.get_evicted_lists is not available")

        # If a dynamic watermark policy is configured, proactively evict some
        # cold lists when we are beyond the high watermark before loading more.
        if getattr(self, "_gpu_memory_policy", "none") == "dynamic_watermark":
            frac = self._current_gpu_list_fraction()
            if self._gpu_watermark_high > 0.0 and frac > self._gpu_watermark_high:
                protected = set(required_list_ids)
                attempts = 0
                # Best-effort: evict at most eviction_max_attempts lists.
                while (
                    frac > self._gpu_watermark_high
                    and attempts < getattr(self, "_eviction_max_attempts", 64)
                ):
                    victim = self._evict_one_list(protected)
                    if victim is None:
                        break
                    attempts += 1
                    frac = self._current_gpu_list_fraction()

        evicted = set(int(x) for x in faiss.get_evicted_lists(self.index))
        to_load = [lid for lid in required_list_ids if lid in evicted]
        if not to_load:
            return
        protected = set(required_list_ids)
        for list_id in to_load:
            self._load_list_with_eviction(list_id, protected)

    def _load_list_with_eviction(self, list_id: int, protected: set) -> None:
        attempts = 0
        while True:
            try:
                bytes_loaded = faiss.load_ivf_lists(self.index, int(list_id))
                if bytes_loaded == 0 and not faiss.is_list_on_gpu(self.index, int(list_id)):
                    raise EvictionPolicyError(
                        f"list {list_id} is not on GPU and cannot be loaded"
                    )
                self._eviction_tracker.record_loaded(int(list_id))
                return
            except Exception as exc:
                if not self._is_oom_error(exc):
                    raise
                evicted = self._evict_one_list(protected)
                if evicted is None:
                    raise EvictionPolicyError(
                        "no eviction candidates available for GPU OOM"
                    ) from exc
                attempts += 1
                if attempts >= self._eviction_max_attempts:
                    raise EvictionPolicyError(
                        f"eviction attempts exceeded while loading list {list_id}"
                    ) from exc

    def _initialize_gpu_ivf_lists(self, config) -> None:
        """Optionally preload a subset of IVF lists to GPU at startup.

        This is a best-effort routine: it walks candidate IVF list IDs in a
        strategy-dependent order and attempts to load them until the GPU memory
        pool can no longer accommodate additional lists.
        """
        strategy = str(getattr(config, "gpu_ivf_init_strategy", "none")).strip().lower()
        if strategy in {"", "none", "disabled"}:
            return
        if not self._supports_gpu_eviction():
            raise EngineError(
                "gpu_ivf_init_strategy requires a GPU IVF index with eviction support"
            )
        if not hasattr(self.index, "nlist"):
            raise EngineError("gpu_ivf_init_strategy requires an IVF index with nlist")

        nlist = int(getattr(self.index, "nlist", 0))
        if nlist <= 0:
            return

        candidate_ids: List[int]
        if strategy == "random":
            candidate_ids = list(range(nlist))
            rng = np.random.default_rng()
            rng.shuffle(candidate_ids)
        elif strategy == "largest":
            try:
                cluster_sizes = self.index.get_cluster_size_heterag(
                    [range(0, nlist)]
                )[0]
            except Exception as exc:
                raise EngineError(
                    "gpu_ivf_init_strategy='largest' requires "
                    "index.get_cluster_size_heterag support"
                ) from exc
            sized = [
                (int(i), int(cluster_sizes[i]))
                for i in range(nlist)
                if int(cluster_sizes[i]) > 0
            ]
            if not sized:
                return
            sized.sort(key=lambda x: x[1], reverse=True)
            candidate_ids = [lid for (lid, _) in sized]
        elif strategy == "frequency":
            stats_path = Path(__file__).resolve().parent / "config" / "default.json"
            try:
                with stats_path.open("r", encoding="utf-8") as f:
                    freq_map = json.load(f)
            except Exception as exc:
                raise EngineError(
                    f"gpu_ivf_init_strategy='frequency' failed to load stats "
                    f"from {stats_path}"
                ) from exc
            items = []
            for k, v in freq_map.items():
                try:
                    lid = int(k)
                    freq = int(v)
                except Exception:
                    continue
                if 0 <= lid < nlist and freq > 0:
                    items.append((lid, freq))
            if not items:
                return
            items.sort(key=lambda x: x[1], reverse=True)
            candidate_ids = [lid for (lid, _) in items]
        else:
            raise EngineError(f"unsupported gpu_ivf_init_strategy: {strategy!r}")

        protected: set = set()
        loaded = 0
        for lid in candidate_ids:
            protected.add(int(lid))
            try:
                self._load_list_with_eviction(int(lid), protected)
                loaded += 1
            except EvictionPolicyError:
                # We reached the practical capacity of the GPU memory pool.
                break
            except Exception as exc:
                raise EngineError(
                    f"failed to initialize IVF list {lid} on GPU"
                ) from exc

    def _evict_one_list(self, protected: set) -> Optional[int]:
        # Prefer backend-driven policy when available.
        policy = getattr(self, "_cluster_policy", None)
        if policy is not None:
            self._refresh_cluster_stats(force=False)
            stats = getattr(self, "_cluster_stats_last_snapshot", {}) or {}
            try:
                victim = policy.select_eviction_candidate(stats, protected)
            except Exception:
                victim = None

            if victim is not None:
                try:
                    if hasattr(faiss, "is_list_on_gpu") and not faiss.is_list_on_gpu(
                        self.index, int(victim)
                    ):
                        victim = None
                    else:
                        faiss.evict_ivf_lists(self.index, int(victim))
                        if getattr(self, "_eviction_tracker", None) is not None:
                            self._eviction_tracker.remove(int(victim))
                        return int(victim)
                except Exception as exc:
                    raise EvictionPolicyError(
                        f"failed to evict IVF list {victim}"
                    ) from exc

        # Fallback to Python-side tracker if configured.
        if getattr(self, "_eviction_tracker", None) is None:
            return None

        while True:
            victim = self._eviction_tracker.pop_victim(protected)
            if victim is None:
                return None
            try:
                if hasattr(faiss, "is_list_on_gpu") and not faiss.is_list_on_gpu(
                    self.index, int(victim)
                ):
                    continue
                faiss.evict_ivf_lists(self.index, int(victim))
                return int(victim)
            except Exception as exc:
                raise EvictionPolicyError(
                    f"failed to evict IVF list {victim}"
                ) from exc

    def evict_lists(self, list_ids: List[int]) -> Dict[int, int]:
        """Manually evict a collection of IVF lists to CPU.

        Returns a mapping from list_id to reclaimed bytes. Raises
        EvictionPolicyError if eviction is not supported.
        """
        if not self._supports_gpu_eviction():
            raise EvictionPolicyError("GPU eviction is not supported by this index")
        if not list_ids:
            return {}
        reclaimed: Dict[int, int] = {}
        for lid in list_ids:
            try:
                bytes_freed = faiss.evict_ivf_lists(self.index, int(lid))
                reclaimed[int(lid)] = int(bytes_freed)
                if getattr(self, "_eviction_tracker", None) is not None:
                    self._eviction_tracker.remove(int(lid))
            except Exception as exc:
                raise EvictionPolicyError(
                    f"failed to evict IVF list {lid}"
                ) from exc
        return reclaimed

    def load_lists(self, list_ids: List[int]) -> Dict[int, int]:
        """Manually load a collection of IVF lists back to GPU.

        Returns a mapping from list_id to loaded bytes. Raises
        EvictionPolicyError if loading fails.
        """
        if not self._supports_gpu_eviction():
            raise EvictionPolicyError("GPU eviction/load is not supported by this index")
        if not list_ids:
            return {}
        loaded: Dict[int, int] = {}
        for lid in list_ids:
            try:
                bytes_loaded = faiss.load_ivf_lists(self.index, int(lid))
                loaded[int(lid)] = int(bytes_loaded)
                if getattr(self, "_eviction_tracker", None) is not None:
                    self._eviction_tracker.record_loaded(int(lid))
            except Exception as exc:
                raise EvictionPolicyError(
                    f"failed to load IVF list {lid} to GPU"
                ) from exc
        return loaded

    @staticmethod
    def _is_oom_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return (
            "out of memory" in msg
            or "cuda error" in msg and "memory" in msg
            or "cudamalloc" in msg
        )
    
    def simulate_onload_cluster(self, config):

        self.cluster_size = self.index.get_cluster_size_heterag([range(0, self.index.nlist)])[0]

        available_gpu_memory = (1 - config.gpu_memory_utilization) * 0.6 * 80
        
        gpu_cluster_memory = 0
        gpu_cluster_to_load = []
        cluster_idx = 0
        cluster_max = 1024
        while (cluster_idx < self.index.nlist and cluster_idx < cluster_max and \
        cluster_idx < len(kv_array) and gpu_cluster_memory < available_gpu_memory):
            cluster_id = kv_array[cluster_idx][0]
            gpu_cluster_memory += self.cluster_size[cluster_id] * self.index.d * 4 / 1024 / 1024 / 1024
            gpu_cluster_to_load.append(cluster_id)
            cluster_idx += 1
        
        print(f"onloading {cluster_idx} clusters")
        self.onload_clusters(gpu_cluster_to_load)
        self.onload_cluster = gpu_cluster_to_load
        self.onload_cluster_hit = 0
        self.total_cluster_search_num = 0

    def _search(self, query: str, num: int = None, return_score=False, eval_cache=False):
        if num is None:
            num = self.topk
        query_emb = self.encoder.encode(query)
        scores, idxs = self.index.search(query_emb, k=num)
        scores = scores.tolist()
        idxs = idxs[0]
        scores = scores[0]

        results = load_docs(self.corpus, idxs)
        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(
        self,
        query_list: List[str],
        query_id_list: Optional[List[SchedulerRequest]] = None,
        num: int = None,
        return_score: bool = False,
        eval_cache: bool = False,
    ):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk
        if query_id_list is None:
            query_id_list = []

        batch_size = self.batch_size

        results = []
        results_emb = EmbeddingInfo()
        scores = []

        encode_time = 0
        search_time = 0
        post_time = 0
        

        for start_idx in range(0, len(query_list), batch_size):

            t1 = time.time()

            query_batch = query_list[start_idx : start_idx + batch_size]
            query_id_batch = query_id_list[start_idx : start_idx + batch_size]

            batch_emb = self.encoder.encode(query_batch)

            t2 = time.time()
            encode_time += t2 - t1

            batch_scores, batch_idxs, cluster_min, cluster_lid = self.index.search_with_cluster_id(batch_emb, k=num)

            if self.return_embedding:
                profile_search_scores, profile_search_idxs = self.index.search(batch_emb, k=20)
                for profile_search_score in profile_search_scores:
                    self.top1.append(profile_search_score[0])
                    self.top5.append(profile_search_score[4])
                    self.top20.append(profile_search_score[19])

            t3 = time.time()
            search_time += t3 - t2

            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()

            flat_idxs = sum(batch_idxs, [])
            batch_results = load_docs(self.corpus, flat_idxs)
            batch_results = [batch_results[i * num : (i + 1) * num] for i in range(len(batch_idxs))]

            scores.extend(batch_scores)
            results.extend(batch_results)


            if (self.return_embedding):
                results_emb.update(query_emb = batch_emb, retrieval_score = batch_scores)

            t4 = time.time()
            post_time += t4 - t3

            centroid_distances, batch_assigned_centroids = self.index.quantizer.search(batch_emb, self.index.nprobe)
            for centroid in batch_assigned_centroids:
                for centroid_id in centroid:
                    if centroid_id not in self.skewness_dict:
                        self.skewness_dict[centroid_id] = 1
                    else:
                        self.skewness_dict[centroid_id] += 1

            if (self.return_embedding):
                if isinstance(self.index, faiss.IndexIVFFlat):

                    centroids = self.index.quantizer.reconstruct_n(0, self.index.nlist)

                    centroid_distances2, batch_assigned_centroids2 = self.index.quantizer.search(batch_emb, self.index.nlist)

                    for taskid, query_str, query_emb, assigned_centroid, batch_data, batch_idx, topk_score, centroid_distance, cluster_min_distance \
                        in zip(query_id_batch, query_list, batch_emb, batch_assigned_centroids, batch_results, batch_idxs, batch_scores, centroid_distances, cluster_min):

                        for centroid in assigned_centroid:
                            self.total_cluster_search_num += 1
                            if centroid in self.onload_cluster:
                                self.onload_cluster_hit += 1

                        last_topk_score = topk_score[-1]

                        # update cluster ranking
                        largest_cluster_ranking = 0
                        for min_cid, min_dist in enumerate(cluster_min_distance):
                            if min_dist <= last_topk_score:
                                largest_cluster_ranking = min_cid
                        self.largest_cluster_ranking.append(largest_cluster_ranking)

                        if not taskid.id in self.request_dict:
                            self.request_dict[taskid.id] = EmbeddingInfo()
                        else:
                            old_assigned_centroid = self.request_dict[taskid.id].centroid_idx[-1]
                            old_centroid_distance = self.request_dict[taskid.id].centroid_distance[-1]

                            # find common centroids
                            common_centroids = np.intersect1d(old_assigned_centroid, assigned_centroid)

                            ordered_common_elements = [x for x in old_assigned_centroid if x in common_centroids]
                            idx_old = [np.where(old_assigned_centroid == x)[0][0] for x in ordered_common_elements]
                            idx_new = [np.where(assigned_centroid == x)[0][0] for x in ordered_common_elements]


                            old_query_emb = self.request_dict[taskid.id].query_emb

                            cid_new_no_overlapped = [x for x in assigned_centroid if x not in common_centroids]
                            cid_new_no_overlapped_id = [i for i, x in enumerate(assigned_centroid) if x not in common_centroids]
                            cdist_test = []
                            for cid in cid_new_no_overlapped:
                                cdist_test.append(fvec_L2sqr(query_emb, centroids[cid]))
                            cdist_test = []
                            for cid in cid_new_no_overlapped:
                                cdist_test.append(fvec_L2sqr(old_query_emb, centroids[cid]))

                            delta_vector = query_emb - self.request_dict[taskid.id].query_emb
                            cangle_test = []
                            
                            for cid in cid_new_no_overlapped:
                                query_centoid_delta = centroids[cid] - old_query_emb
                                cangle_test.append(fvec_inner_product(delta_vector[0], query_centoid_delta[0]))

                            cdist_test = []
                            for cid in cid_new_no_overlapped_id:
                                cdist_test.append(cluster_min_distance[cid])

                            dist_old = old_centroid_distance[idx_old]
                            dist_new = centroid_distance[idx_new]

                            
                            diff_1 = np.array(dist_new) - np.array(dist_old)
                            diff_2 = np.array(centroid_distance)[idx_new] - np.array(old_centroid_distance)[idx_old]
        
                            mask_1 = diff_1 > 0
                            mask_2 = diff_2 > 0

                            self.total_iter_clusters += len(old_centroid_distance)
                            self.total_overlap_clusters += len(mask_1)
                            self.total_far_clusters += np.sum(mask_1)
                            self.total_far_dist += np.sum(mask_1 & mask_2)

                            new_old_query_dist = fvec_L2sqr(np.array(query_emb), np.array(old_query_emb))


                            target_cluster = []
                            for min_cid, min_dist in enumerate(cluster_min_distance):
                                if min_dist <= last_topk_score:
                                    target_cluster.append(assigned_centroid[min_cid])
                            overlapped_useful_cluster_number = 0
                            for cluster in target_cluster:
                                if cluster in old_assigned_centroid:
                                    overlapped_useful_cluster_number += 1
                            in_old_centroid = False
                            if overlapped_useful_cluster_number == len(target_cluster):
                                in_old_centroid = True
                                self.in_old_centroid_set += 1
                                self.in_old_centroid_set_list.append(new_old_query_dist)
                            self.overlapped_rate.append(overlapped_useful_cluster_number / len(target_cluster))
                            

                            old_batch_idx = self.request_dict[taskid.id].doc_idx
                            old_flat_idxs = sum(old_batch_idx, [])
                            old_batch_results = load_docs(self.corpus, old_flat_idxs)
                            old_doc = []
                            for doc in old_batch_results:
                                old_doc.append(get_content(doc))

                            self.total_repeat_search += num
                            answer_in_doc = False
                            answer_in_query = False

                            old_search_scores, old_search_idxs = self.index.search(np.array(old_query_emb), k=20)
                            in_old_query_top20 = 0
                            for the_idx in batch_idx:
                                if the_idx in old_search_idxs:
                                    in_old_query_top20 += 1
                                    self.answer_in_old_query_search += 1
                                    self.answer_in_old_query_search_list.append(fvec_L2sqr(np.array(query_emb),
                                    np.array(old_query_emb)))
                                else:
                                    self.answer_not_in_old_query_search_list.append(fvec_L2sqr(np.array(query_emb),
                                    np.array(old_query_emb)))

                            if in_old_query_top20 == num:
                                answer_in_query = True
        
                            old_dist_debug, old_centroid_debug = self.index.quantizer.search(np.array(old_query_emb), self.index.nprobe)
                            old_search_scores, old_search_idxs, old_cluster_min, old_cluster_lid = self.index.search_with_cluster_id(np.array(old_query_emb), k=self.larger_topk)

                            last_topk_score = old_search_scores[0][-1]
                            old_assigned_centroid_top20 = []
                            for old_cluster_min_distance in old_cluster_min:
                                for min_cid, min_dist in enumerate(old_cluster_min_distance):
                                    if min_dist <= last_topk_score:
                                        old_assigned_centroid_top20.append(old_assigned_centroid[min_cid])
                            overlapped_useful_cluster_number = 0
                            for cluster in target_cluster:
                                if cluster in old_assigned_centroid_top20:
                                    overlapped_useful_cluster_number += 1

                            in_old_centroid_top20 = False
                            if overlapped_useful_cluster_number == len(target_cluster):
                                in_old_centroid_top20 = True
                                self.in_old_hit_centroid_top20 += 1
                                self.in_old_hit_centroid_top20_list.append(new_old_query_dist)
                            self.overlapped_rate_top20.append(overlapped_useful_cluster_number / len(target_cluster))

                            # old doc top20
                            in_old_doc_top20 = 0
                            for the_idx in batch_idx:
                                if the_idx in old_search_idxs:
                                    in_old_doc_top20 += 1
                                    self.answer_in_doc_search += 1
                                    self.answer_in_doc_search_list.append((1, largest_cluster_ranking))
                            if in_old_doc_top20 == len(batch_idx):
                                answer_in_doc = True
                                self.all_answer_in_doc += 1
                                self.all_answer_in_doc_list.append(new_old_query_dist)

                            
                            if in_old_centroid and not answer_in_doc and not answer_in_query:
                                self.in_old_centroid_but_not_in_local_buffer += 1
                            if in_old_centroid and (answer_in_doc or answer_in_query):
                                self.in_old_centroid_and_in_local_buffer += 1
                            if in_old_centroid_top20 and not answer_in_doc and not answer_in_query:
                                self.in_old_centroid_top20_but_not_in_local_buffer += 1
                            if in_old_centroid_top20 and (answer_in_doc or answer_in_query):
                                self.in_old_centroid_top20_and_in_local_buffer += 1
                            
                            if answer_in_doc and answer_in_query:
                                self.answer_in_both += 1

                            self.old_termination_point.append(largest_cluster_ranking + 1)
                            if answer_in_doc:
                                self.new_termination_point.append(0)
                                self.new_termination_point2.append(0)
                                self.new_termination_point3.append(0)
                            else:
                                self.new_termination_point3.append(largest_cluster_ranking + 1)

                                et_hit_centroid = set(old_assigned_centroid_top20)
                                et_nohit_centroid = set(old_assigned_centroid)
                                indices = np.arange(len(assigned_centroid))
                                sorted_indices = sorted(indices, key=lambda i: (0 if assigned_centroid[i] in et_hit_centroid else 1 if assigned_centroid[i] in et_nohit_centroid else 2))
                                sorted_assigned_centroid = assigned_centroid[sorted_indices]


                                sorted_largest = 0
                                for i, cluster_id in enumerate(sorted_assigned_centroid):
                                    if cluster_id in target_cluster:
                                        sorted_largest = i
                                self.new_termination_point.append(sorted_largest + 1)

                                sorted_indices = sorted(indices, key=lambda i: (0 if assigned_centroid[i] in et_hit_centroid else 1))
                                sorted_assigned_centroid = assigned_centroid[sorted_indices]

                                sorted_largest = 0
                                for i, cluster_id in enumerate(sorted_assigned_centroid):
                                    if cluster_id in target_cluster:
                                        sorted_largest = i
                                if new_old_query_dist < 0.3:
                                    self.new_termination_point2.append(sorted_largest + 1)
                                else:
                                    self.new_termination_point2.append(largest_cluster_ranking + 1)

                        self.request_dict[taskid.id].update(query_emb = [query_emb], 
                        centroid_idx = [assigned_centroid], 
                        centroid_distance = [centroid_distance], 
                        topk_score = [topk_score], 
                        largest_cluster = [largest_cluster_ranking],
                        doc_idx = [batch_idx])

            if (eval_cache):
                for batch_idx in batch_idxs:
                    for idxs in batch_idx[0:1]:
                        if (idxs in self.idxs_cache):
                            self.idxs_cache[idxs] += 1
                        else:
                            self.idxs_cache[idxs] = 1

        if return_score:
            return results, scores
        else:
            return results
    
    def show_inter_diff(self):
        inter_dis = []
        for taskid, request in self.request_dict.items():
            inter_dis.append(request.show_inter_stage_diff(taskid, metric = 1))
        print("[similarity] average inter dis", np.mean(inter_dis))
        print("[similarity] average top1", np.mean(self.top1))
        print("[similarity] average top5", np.mean(self.top5))
        print("[similarity] average top20", np.mean(self.top20))

        print("average overlap rate", np.mean(self.overlapped_rate))
        print("average overlap rate top20", np.mean(self.overlapped_rate_top20))
        count = np.sum(np.array(inter_dis) < 0.15)
        print(f"< 0.15, {count}")

    def show_time_profile(self):
        print("Not implemented")