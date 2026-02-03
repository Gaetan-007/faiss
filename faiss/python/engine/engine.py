# generator initialization built upon FlashRAG
from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict, deque
from concurrent.futures import Future
from dataclasses import dataclass
from ..utils import *
from .encoder import Encoder, STEncoder
from typing import List, Dict, Optional
import threading
import uuid
import faiss
import time
import numpy as np
from heapq import heappush, heappop
from .scheduler import BaseScheduler, SchedulerRequest, FifoScheduler


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
        if policy_norm not in {"lru", "fifo"}:
            raise EvictionPolicyError(
                f"unsupported eviction policy: {policy}. Use 'lru' or 'fifo'"
            )
        self._policy = policy_norm
        self._lru = OrderedDict() if policy_norm == "lru" else None
        self._fifo_queue = deque() if policy_norm == "fifo" else None
        self._fifo_set = set() if policy_norm == "fifo" else None

    def record_access(self, list_ids: List[int]) -> None:
        for list_id in list_ids:
            self.record_loaded(list_id)

    def record_loaded(self, list_id: int) -> None:
        if self._policy == "lru":
            self._lru.pop(list_id, None)
            self._lru[list_id] = None
        else:
            if list_id in self._fifo_set:
                return
            self._fifo_queue.append(list_id)
            self._fifo_set.add(list_id)

    def remove(self, list_id: int) -> None:
        if self._policy == "lru":
            self._lru.pop(list_id, None)
        else:
            if list_id in self._fifo_set:
                self._fifo_set.remove(list_id)

    def pop_victim(self, protected: set) -> Optional[int]:
        if self._policy == "lru":
            for list_id in list(self._lru.keys()):
                if list_id in protected:
                    continue
                self._lru.pop(list_id, None)
                return list_id
            return None

        while self._fifo_queue:
            list_id = self._fifo_queue.popleft()
            if list_id not in self._fifo_set:
                continue
            self._fifo_set.remove(list_id)
            if list_id in protected:
                continue
            return list_id
        return None


class FaissEnginConfig(BaseConfig):
    """Config for all engines."""
    index_type: str
    index_path: str
    corpus_path: str

    retrieval_method: str
    retrieval_topk: int
    retrieval_batch_size: int
    retrieval_model_path: Optional[str] = None
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
        
        self.reinit(config)

        if getattr(config, "async_enabled", False):
            scheduler = getattr(config, "async_scheduler", None)
            if scheduler is None:
                scheduler = FifoScheduler(
                    max_batch_size=getattr(config, "async_max_batch_size", self.batch_size),
                    max_wait_ms=getattr(config, "async_max_wait_ms", 0),
                    step_interval_ms=getattr(config, "async_step_interval_ms", 0),
                )
            if not isinstance(scheduler, BaseScheduler):
                raise EngineError("async_scheduler must be a BaseScheduler instance")
            self.enable_async(
                scheduler=scheduler,
                max_queue_size=getattr(config, "async_queue_size", 1024),
                idle_sleep_s=getattr(config, "async_idle_sleep_s", 0.01),
            )
    
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

        # self.simulate_onload_cluster(config)

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
                faiss.set_auto_fetch(self.index, False)
            except Exception as exc:
                raise EvictionPolicyError(
                    "failed to disable auto-fetch before eviction control"
                ) from exc

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

    def _maybe_prefetch_lists(self, query_emb: np.ndarray) -> None:
        if not getattr(self, "_eviction_enabled", False):
            return
        list_ids = self._compute_required_list_ids(query_emb)
        self._ensure_lists_on_gpu(list_ids)
        self._eviction_tracker.record_access(list_ids)

    def _compute_required_list_ids(self, query_emb: np.ndarray) -> List[int]:
        if not hasattr(self.index, "quantizer"):
            raise EvictionPolicyError("index does not expose quantizer for prefetch")
        if not hasattr(self.index, "nprobe"):
            raise EvictionPolicyError("index does not expose nprobe for prefetch")
        _, list_ids = self.index.quantizer.search(query_emb, int(self.index.nprobe))
        list_ids = np.unique(list_ids.reshape(-1))
        return [int(lid) for lid in list_ids if lid >= 0]

    def _ensure_lists_on_gpu(self, required_list_ids: List[int]) -> None:
        if not required_list_ids:
            return
        if not hasattr(faiss, "get_evicted_lists"):
            raise EvictionPolicyError("faiss.get_evicted_lists is not available")
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

    def _evict_one_list(self, protected: set) -> Optional[int]:
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

            self._maybe_prefetch_lists(batch_emb)

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