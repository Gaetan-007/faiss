import numpy as np
import pytest

import faiss

from faiss.engine.engine import FaissEngine, FaissEnginConfig


class _ConcreteFaissEngine(FaissEngine):
    """Concrete subclass for tests; FaissEngine is abstract (_add/_batch_add not implemented)."""

    def _add(self, query: str, return_centroid: bool, retrain: bool):
        raise NotImplementedError("test double")

    def _batch_add(self, query_list, return_centroid: bool, retrain: bool):
        raise NotImplementedError("test double")


def _make_config(**overrides):
    base = dict(
        index_type="IVFFlat",
        index_path="/tmp/unused.faiss",
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
        eviction_policy="none",
        eviction_max_attempts=4,
        nprobe=8,
    )
    base.update(overrides)
    return FaissEnginConfig(**base)


def _build_cpu_ivfflat(d=32, nb=6000, nlist=128, nprobe=8):
    rng = np.random.RandomState(123)
    xb = rng.random((nb, d)).astype("float32")
    xq = rng.random((50, d)).astype("float32")
    quantizer = faiss.IndexFlatL2(d)
    cpu_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    cpu_index.train(xb)
    cpu_index.add(xb)
    cpu_index.nprobe = nprobe
    return cpu_index, xb, xq


@pytest.mark.skipif(faiss.get_num_gpus() < 2, reason="requires >= 2 GPUs")
def test_engine_builds_multi_gpu_sharded_index():
    cpu_index, xb, xq = _build_cpu_ivfflat()

    cfg = _make_config(
        gpu_enabled=True,
        gpu_ngpu=2,
        gpu_devices=[0, 1],
        gpu_shard=True,
        gpu_shard_type=5,
        gpu_common_ivf_quantizer=True,
        gpu_miss_policy="error",
    )

    engine = _ConcreteFaissEngine.__new__(_ConcreteFaissEngine)
    engine.config = cfg
    engine.index = cpu_index
    engine._cpu_index = cpu_index
    engine._gpu_resources = None

    engine._maybe_build_gpu_index(cfg)
    assert hasattr(engine.index, "count") and hasattr(engine.index, "at")
    assert int(engine.index.count()) == 2


@pytest.mark.skipif(faiss.get_num_gpus() < 2, reason="requires >= 2 GPUs")
def test_engine_propagates_miss_policy_to_shards_autofetch():
    if not hasattr(faiss, "set_auto_fetch"):
        pytest.skip("auto-fetch helpers not available in this build")

    cpu_index, xb, xq = _build_cpu_ivfflat()
    cfg = _make_config(
        gpu_enabled=True,
        gpu_ngpu=2,
        gpu_devices=[0, 1],
        gpu_shard=True,
        gpu_shard_type=4,
        gpu_common_ivf_quantizer=True,
        gpu_miss_policy="auto_fetch",
        eviction_policy="none",
    )

    engine = _ConcreteFaissEngine.__new__(_ConcreteFaissEngine)
    engine.config = cfg
    engine.index = cpu_index
    engine._cpu_index = cpu_index
    engine._gpu_resources = None
    engine._eviction_enabled = False

    engine._maybe_build_gpu_index(cfg)
    engine._configure_gpu_miss_policy(cfg)

    for s in range(int(engine.index.count())):
        shard = faiss.downcast_index(engine.index.at(s))
        assert faiss.is_auto_fetch_enabled(shard)


@pytest.mark.skipif(faiss.get_num_gpus() < 2, reason="requires >= 2 GPUs")
def test_engine_sharded_autofetch_restores_after_evict():
    if not all(
        hasattr(faiss, name)
        for name in ("set_auto_fetch", "reset_auto_fetch_stats", "get_auto_fetch_stats")
    ):
        pytest.skip("auto-fetch helpers not available in this build")

    cpu_index, xb, xq = _build_cpu_ivfflat(nb=8000, nlist=128, nprobe=8)
    k = 5
    D_ref, I_ref = cpu_index.search(xq, k)

    cfg = _make_config(
        gpu_enabled=True,
        gpu_ngpu=2,
        gpu_devices=[0, 1],
        gpu_shard=True,
        gpu_shard_type=4,
        gpu_common_ivf_quantizer=True,
        gpu_miss_policy="auto_fetch",
        eviction_policy="none",
        nprobe=8,
    )

    engine = _ConcreteFaissEngine.__new__(_ConcreteFaissEngine)
    engine.config = cfg
    engine.index = cpu_index
    engine._cpu_index = cpu_index
    engine._gpu_resources = None
    engine._eviction_enabled = False

    engine._maybe_build_gpu_index(cfg)
    engine._configure_gpu_miss_policy(cfg)

    # Evict all lists touched by the workload.
    _, list_ids = cpu_index.quantizer.search(xq, cpu_index.nprobe)
    touched = np.unique(list_ids.reshape(-1))
    touched = [int(i) for i in touched if cpu_index.invlists.list_size(int(i)) > 0]
    if not touched:
        pytest.skip("No non-empty lists touched for eviction")

    faiss.evict_ivf_lists(engine.index, touched)

    # Search should transparently auto-fetch and match CPU baseline.
    D_gpu, I_gpu = engine.index.search(xq, k)
    np.testing.assert_array_equal(I_ref, I_gpu)
    np.testing.assert_array_almost_equal(D_ref, D_gpu, decimal=4)

    total_fetches = 0
    for s in range(int(engine.index.count())):
        shard = faiss.downcast_index(engine.index.at(s))
        stats = faiss.get_auto_fetch_stats(shard)
        total_fetches += int(stats.get("total_fetches", 0))
    assert total_fetches >= 1


@pytest.mark.skipif(faiss.get_num_gpus() < 2, reason="requires >= 2 GPUs")
def test_engine_builds_cpu_offload_sharded_index_and_searches():
    if not hasattr(faiss, "IvfListMissPolicy_CpuOffload"):
        pytest.skip("CpuOffload miss policy not available")
    if not hasattr(faiss, "build_sharded_ivfflat_cpu_offload_index"):
        pytest.skip("CpuOffload sharded builder not available")

    cpu_index, xb, xq = _build_cpu_ivfflat(nb=8000, nlist=128, nprobe=8)
    k = 5
    D_ref, I_ref = cpu_index.search(xq, k)

    cfg = _make_config(
        gpu_enabled=True,
        gpu_ngpu=2,
        gpu_devices=[0, 1],
        gpu_shard=True,
        gpu_shard_type=5,
        gpu_common_ivf_quantizer=True,
        gpu_miss_policy="cpu_offload",
        eviction_policy="none",
        nprobe=8,
    )

    engine = _ConcreteFaissEngine.__new__(_ConcreteFaissEngine)
    engine.config = cfg
    engine.index = cpu_index
    engine._cpu_index = cpu_index
    engine._gpu_resources = None
    engine._eviction_enabled = False

    engine._maybe_build_gpu_index(cfg)
    engine._configure_gpu_miss_policy(cfg)

    D_gpu, I_gpu = engine.index.search(xq, k)
    np.testing.assert_array_equal(I_ref, I_gpu)
    np.testing.assert_allclose(D_ref, D_gpu, rtol=1e-6, atol=1e-3)

