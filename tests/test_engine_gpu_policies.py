import numpy as np
import pytest

import faiss
from faiss.engine.engine import FaissEngine, FaissEnginConfig, _EvictionTracker


class _ConcreteFaissEngine(FaissEngine):
    """Concrete subclass for tests; FaissEngine is abstract (_add/_batch_add not implemented)."""

    def _add(self, query: str, return_centroid: bool, retrain: bool):
        raise NotImplementedError("test double")

    def _batch_add(self, query_list, return_centroid: bool, retrain: bool):
        raise NotImplementedError("test double")


class _DummyEncoder:
    def __init__(self, dim: int):
        self._dim = dim

    def encode(self, query):
        if isinstance(query, str):
            return np.random.rand(1, self._dim).astype("float32")
        return np.random.rand(len(query), self._dim).astype("float32")


class _DummyConfig(FaissEnginConfig):
    # Helper config subclass that allows minimal construction in tests
    pass


def _make_small_gpu_ivfflat_index(d=16, nb=1000, nlist=10):
    xb = np.random.rand(nb, d).astype("float32")
    quantizer = faiss.IndexFlatL2(d)
    cpu_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    cpu_index.train(xb)
    cpu_index.add(xb)
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    return gpu_index, xb


@pytest.mark.skipif(faiss.get_num_gpus() < 1, reason="gpu only test")
def test_engine_configures_gpu_miss_policy_error():
    gpu_index, xb = _make_small_gpu_ivfflat_index()
    index_path = "/tmp/test_engine_gpu_index.faiss"
    faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), index_path)

    cfg = FaissEnginConfig(
        index_type="IVFFlat",
        index_path=index_path,
        corpus_path="imdb",  # not used in this test
        retrieval_method="dummy",
        retrieval_topk=5,
        retrieval_batch_size=4,
        retrieval_model_path="dummy",
        retrieval_query_max_length=8,
        retrieval_use_fp16=False,
        retrieval_pooling_method="mean",
        return_embedding=False,
        larger_topk=5,
        use_sentence_transformer=False,
        gpu_memory_utilization=0.8,
        eviction_policy="lru",
        eviction_max_attempts=4,
        nprobe=1,
    )

    engine = _ConcreteFaissEngine.__new__(_ConcreteFaissEngine)
    engine.config = cfg
    # Bypass heavy encoder / corpus initialization in tests
    engine.index = gpu_index
    engine.corpus = [None] * xb.shape[0]
    engine.encoder = _DummyEncoder(dim=gpu_index.d)

    # Configure miss policy directly; should not raise for 'error'
    engine._configure_gpu_miss_policy(cfg)


@pytest.mark.skipif(faiss.get_num_gpus() < 1, reason="gpu only test")
def test_engine_dynamic_watermark_policy_does_not_crash():
    gpu_index, xb = _make_small_gpu_ivfflat_index()
    index_path = "/tmp/test_engine_gpu_index2.faiss"
    faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), index_path)

    cfg = FaissEnginConfig(
        index_type="IVFFlat",
        index_path=index_path,
        corpus_path="imdb",
        retrieval_method="dummy",
        retrieval_topk=5,
        retrieval_batch_size=4,
        retrieval_model_path="dummy",
        retrieval_query_max_length=8,
        retrieval_use_fp16=False,
        retrieval_pooling_method="mean",
        return_embedding=False,
        larger_topk=5,
        use_sentence_transformer=False,
        gpu_memory_utilization=0.8,
        eviction_policy="lru",
        eviction_max_attempts=4,
        nprobe=1,
        gpu_memory_policy="dynamic_watermark",
        gpu_watermark_high=0.9,
    )

    engine = _ConcreteFaissEngine.__new__(_ConcreteFaissEngine)
    engine.config = cfg
    engine.index = gpu_index
    engine.corpus = [None] * xb.shape[0]
    engine.encoder = _DummyEncoder(dim=gpu_index.d)
    engine._eviction_enabled = True
    engine._eviction_tracker = _EvictionTracker("lru")
    engine._eviction_max_attempts = 4

    engine._configure_gpu_memory_policy(cfg)


@pytest.mark.skipif(faiss.get_num_gpus() < 1, reason="gpu only test")
def test_backend_activation_stats_increase_on_search():
    """Backend IVF activation stats should increment with repeated searches."""
    if not hasattr(faiss, "get_list_activation_stats"):
        pytest.skip("get_list_activation_stats not available in this build")

    gpu_index, xb = _make_small_gpu_ivfflat_index()
    xq = xb[:5]

    stats_before = faiss.get_list_activation_stats(gpu_index)
    before_map = {
        int(s["list_id"]): int(s.get("probe_count", 0)) for s in stats_before
    } if stats_before is not None else {}

    # Issue multiple searches to touch IVF lists.
    gpu_index.search(xq, 5)
    gpu_index.search(xq, 5)

    stats_after = faiss.get_list_activation_stats(gpu_index)
    assert stats_after is not None
    after_map = {
        int(s["list_id"]): int(s.get("probe_count", 0)) for s in stats_after
    }

    # At least one list should have a strictly larger probe_count.
    increased = any(
        after_map.get(lid, 0) > before_map.get(lid, 0) for lid in after_map.keys()
    )
    assert increased


@pytest.mark.skipif(faiss.get_num_gpus() < 1, reason="gpu only test")
def test_backend_activation_stats_reset():
    """reset_list_activation_stats should zero out probe counters."""
    if not hasattr(faiss, "get_list_activation_stats"):
        pytest.skip("get_list_activation_stats not available in this build")
    if not hasattr(faiss, "reset_list_activation_stats"):
        pytest.skip("reset_list_activation_stats not available in this build")

    gpu_index, xb = _make_small_gpu_ivfflat_index()
    xq = xb[:5]

    gpu_index.search(xq, 5)
    stats = faiss.get_list_activation_stats(gpu_index)
    assert stats is not None
    assert any(int(s.get("probe_count", 0)) > 0 for s in stats)

    faiss.reset_list_activation_stats(gpu_index)
    stats_reset = faiss.get_list_activation_stats(gpu_index)
    assert stats_reset is not None
    assert all(int(s.get("probe_count", 0)) == 0 for s in stats_reset)

