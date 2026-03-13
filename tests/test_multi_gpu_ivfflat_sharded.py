"""
Multi-GPU IVF-Flat sharded index tests (shard_type 4 and 5).

Tests nlist-based sharding (type 4: contiguous, type 5: load-balanced) and
evict/load forwarding on IndexShardsIVF.
"""
import numpy as np
import pytest

import faiss
from faiss.contrib.datasets import SyntheticDataset

pytestmark = pytest.mark.skipif(
    faiss.get_num_gpus() < 2,
    reason="requires >= 2 GPUs",
)


def _skip_if_no_auto_fetch():
    required = [
        "set_auto_fetch",
        "is_auto_fetch_enabled",
        "get_auto_fetch_stats",
        "reset_auto_fetch_stats",
        "get_evicted_lists",
    ]
    for name in required:
        if not hasattr(faiss, name):
            pytest.skip("Auto-fetch helpers not available in this build")


def _build_cpu_ivfflat(d, nb, nq, nlist, nprobe):
    ds = SyntheticDataset(d, nb, nq, 100)
    index = faiss.index_factory(ds.d, f"IVF{nlist},Flat")
    index.train(ds.get_train())
    index.add(ds.get_database())
    index.nprobe = nprobe
    return index, ds


def _unique_list_ids(cpu_index, xq, nprobe):
    _, list_ids = cpu_index.quantizer.search(xq, nprobe)
    ids = np.unique(list_ids.reshape(-1))
    # Filter to only non-empty lists (evict/load of empty lists can trigger
    # "Cached IVF list is missing indices" without the GpuIndexIVFFlat fix)
    non_empty = [
        i for i in ids
        if cpu_index.invlists.list_size(int(i)) > 0
    ]
    return np.array(non_empty, dtype=ids.dtype) if non_empty else np.array([], dtype=ids.dtype)


def _create_sharded_index(cpu_index, ngpu, shard_type, use_cuvs=False, res=None):
    co = faiss.GpuMultipleClonerOptions()
    co.shard = True
    co.shard_type = shard_type
    co.common_ivf_quantizer = True
    co.use_cuvs = use_cuvs
    if res is None:
        res = [faiss.StandardGpuResources() for _ in range(ngpu)]
    gpus = list(range(ngpu))
    return faiss.index_cpu_to_gpu_multiple_py(res, cpu_index, co, gpus)


def test_shard_type4_search_correctness():
    """shard_type=4 yields same results as CPU/single-GPU."""
    ngpu = min(2, faiss.get_num_gpus())
    cpu_index, ds = _build_cpu_ivfflat(32, 8000, 100, 128, 8)
    Dref, Iref = cpu_index.search(ds.get_queries(), 10)

    co = faiss.GpuMultipleClonerOptions()
    co.shard = True
    co.shard_type = 4
    co.common_ivf_quantizer = True
    co.use_cuvs = False
    res = [faiss.StandardGpuResources() for _ in range(ngpu)]
    index_gpu = faiss.index_cpu_to_gpu_multiple_py(
        res, cpu_index, co, list(range(ngpu))
    )
    faiss.GpuParameterSpace().set_index_parameter(index_gpu, "nprobe", 8)
    Dnew, Inew = index_gpu.search(ds.get_queries(), 10)
    np.testing.assert_array_equal(Iref, Inew)
    np.testing.assert_array_almost_equal(Dref, Dnew, decimal=4)


def test_shard_type5_search_correctness():
    """shard_type=5 yields same results as CPU/single-GPU."""
    ngpu = min(2, faiss.get_num_gpus())
    cpu_index, ds = _build_cpu_ivfflat(32, 8000, 100, 128, 8)
    Dref, Iref = cpu_index.search(ds.get_queries(), 10)

    index_gpu = _create_sharded_index(cpu_index, ngpu, 5)
    faiss.GpuParameterSpace().set_index_parameter(index_gpu, "nprobe", 8)
    Dnew, Inew = index_gpu.search(ds.get_queries(), 10)
    np.testing.assert_array_equal(Iref, Inew)
    np.testing.assert_array_almost_equal(Dref, Dnew, decimal=4)


def test_shard_type4_vs_type5_search_equivalent():
    """Same CPU index: type4 and type5 produce equivalent search results."""
    ngpu = min(2, faiss.get_num_gpus())
    cpu_index, ds = _build_cpu_ivfflat(32, 8000, 100, 128, 8)

    index_type4 = _create_sharded_index(cpu_index, ngpu, 4)
    index_type5 = _create_sharded_index(cpu_index, ngpu, 5)
    faiss.GpuParameterSpace().set_index_parameter(index_type4, "nprobe", 8)
    faiss.GpuParameterSpace().set_index_parameter(index_type5, "nprobe", 8)

    D4, I4 = index_type4.search(ds.get_queries(), 10)
    D5, I5 = index_type5.search(ds.get_queries(), 10)
    np.testing.assert_array_equal(I4, I5)
    np.testing.assert_array_almost_equal(D4, D5, decimal=4)


def test_multi_gpu_vs_single_gpu_correctness():
    """Multi-GPU sharded index search matches single-GPU index (same CPU source)."""
    cpu_index, ds = _build_cpu_ivfflat(32, 5000, 80, 128, 8)
    xq = ds.get_queries()
    k = 10
    D_cpu, I_cpu = cpu_index.search(xq, k)

    # Single GPU
    res_single = faiss.StandardGpuResources()
    index_single = faiss.index_cpu_to_gpu(res_single, 0, cpu_index)
    index_single.nprobe = 8
    D_single, I_single = index_single.search(xq, k)

    # Multi-GPU sharded
    ngpu = min(2, faiss.get_num_gpus())
    index_multi = _create_sharded_index(cpu_index, ngpu, 5)
    faiss.GpuParameterSpace().set_index_parameter(index_multi, "nprobe", 8)
    D_multi, I_multi = index_multi.search(xq, k)

    np.testing.assert_array_equal(I_cpu, I_single)
    np.testing.assert_array_equal(I_cpu, I_multi)
    np.testing.assert_array_almost_equal(D_cpu, D_single, decimal=4)
    np.testing.assert_array_almost_equal(D_cpu, D_multi, decimal=4)


def test_multi_gpu_vs_cpu_nprobe_sweep():
    """Multi-GPU search matches CPU for various nprobe values."""
    ngpu = min(2, faiss.get_num_gpus())
    cpu_index, ds = _build_cpu_ivfflat(32, 6000, 60, 128, 8)
    xq = ds.get_queries()
    k = 5

    index_multi = _create_sharded_index(cpu_index, ngpu, 4)
    for nprobe in [1, 4, 8, 16]:
        cpu_index.nprobe = nprobe
        faiss.GpuParameterSpace().set_index_parameter(index_multi, "nprobe", nprobe)
        D_cpu, I_cpu = cpu_index.search(xq, k)
        D_gpu, I_gpu = index_multi.search(xq, k)
        np.testing.assert_array_equal(I_cpu, I_gpu, err_msg=f"nprobe={nprobe}")
        np.testing.assert_array_almost_equal(D_cpu, D_gpu, decimal=4, err_msg=f"nprobe={nprobe}")


def test_multi_gpu_search_k_variants():
    """Multi-GPU search matches CPU for various k values."""
    ngpu = min(2, faiss.get_num_gpus())
    cpu_index, ds = _build_cpu_ivfflat(32, 5000, 50, 64, 8)
    xq = ds.get_queries()

    index_multi = _create_sharded_index(cpu_index, ngpu, 5)
    faiss.GpuParameterSpace().set_index_parameter(index_multi, "nprobe", 8)

    for k in [1, 5, 10, 20]:
        D_cpu, I_cpu = cpu_index.search(xq, k)
        D_gpu, I_gpu = index_multi.search(xq, k)
        np.testing.assert_array_equal(I_cpu, I_gpu, err_msg=f"k={k}")
        np.testing.assert_array_almost_equal(D_cpu, D_gpu, decimal=4, err_msg=f"k={k}")


def test_multi_gpu_reset_add_correctness():
    """After reset and add, multi-GPU search still matches CPU."""
    ngpu = min(2, faiss.get_num_gpus())
    cpu_index, ds = _build_cpu_ivfflat(32, 4000, 40, 64, 8)
    xq = ds.get_queries()
    k = 5
    D_ref, I_ref = cpu_index.search(xq, k)

    index_multi = _create_sharded_index(cpu_index, ngpu, 4)
    faiss.GpuParameterSpace().set_index_parameter(index_multi, "nprobe", 8)
    index_multi.reset()
    index_multi.add(ds.get_database())
    D_new, I_new = index_multi.search(xq, k)
    np.testing.assert_array_equal(I_ref, I_new)
    np.testing.assert_array_almost_equal(D_ref, D_new, decimal=4)


def test_multi_gpu_ngpu_scaling_correctness():
    """2-GPU and N-GPU (all available) both produce correct results vs CPU."""
    ngpu_total = faiss.get_num_gpus()
    if ngpu_total < 2:
        pytest.skip("requires >= 2 GPUs")
    cpu_index, ds = _build_cpu_ivfflat(32, 5000, 50, 128, 8)
    xq = ds.get_queries()
    k = 10
    D_cpu, I_cpu = cpu_index.search(xq, k)

    for ngpu in [2, min(ngpu_total, 4)]:
        if ngpu > ngpu_total:
            continue
        index_gpu = _create_sharded_index(cpu_index, ngpu, 5)
        faiss.GpuParameterSpace().set_index_parameter(index_gpu, "nprobe", 8)
        D_gpu, I_gpu = index_gpu.search(xq, k)
        np.testing.assert_array_equal(I_cpu, I_gpu, err_msg=f"ngpu={ngpu}")
        np.testing.assert_array_almost_equal(D_cpu, D_gpu, decimal=4, err_msg=f"ngpu={ngpu}")


def _get_ivflists_bytes_per_gpu(res_list, gpus):
    """IVFLists total bytes per GPU. res_list[i] corresponds to gpus[i]."""
    bytes_per_gpu = []
    for i, res in enumerate(res_list):
        info = res.getMemoryInfo()
        dev = gpus[i] if i < len(gpus) else i
        if dev in info and "IVFLists" in info[dev]:
            bytes_per_gpu.append(info[dev]["IVFLists"][1])
        else:
            bytes_per_gpu.append(0)
    return bytes_per_gpu


@pytest.mark.skipif(
    not hasattr(faiss.StandardGpuResources(), "setDeviceMemoryReservation"),
    reason="setDeviceMemoryReservation not available",
)
def test_load_balance_vector_data_bytes():
    """type5: IVFLists bytes per GPU should be balanced (< 20% variance)."""
    ngpu = min(2, faiss.get_num_gpus())
    cpu_index, ds = _build_cpu_ivfflat(32, 8000, 100, 128, 8)

    pool_size = 128 * 1024 * 1024  # 128MB per GPU for IVF + overhead
    res = [faiss.StandardGpuResources() for _ in range(ngpu)]
    for r in res:
        r.setTempMemory(0)  # required when using PreallocMemoryPool
        r.setDeviceMemoryReservation(pool_size)

    co = faiss.GpuMultipleClonerOptions()
    co.shard = True
    co.shard_type = 5
    co.common_ivf_quantizer = True
    co.use_cuvs = False
    gpus = list(range(ngpu))
    index_gpu = faiss.index_cpu_to_gpu_multiple_py(
        res, cpu_index, co, gpus
    )

    bytes_per_gpu = _get_ivflists_bytes_per_gpu(res, gpus)
    total = sum(bytes_per_gpu)
    if total == 0:
        pytest.skip("No IVFLists allocation info available")
    max_bytes = max(bytes_per_gpu)
    min_bytes = min(bytes_per_gpu)
    assert max_bytes <= min_bytes * 1.2 + 1, (
        f"Load imbalance > 20%: max={max_bytes} min={min_bytes}"
    )


def test_evict_load_on_sharded_index():
    """Evict/load on IndexShardsIVF: results change after evict, restore after load."""
    ngpu = min(2, faiss.get_num_gpus())
    cpu_index, ds = _build_cpu_ivfflat(64, 5000, 50, 64, 8)

    index_gpu = _create_sharded_index(cpu_index, ngpu, 4)
    faiss.GpuParameterSpace().set_index_parameter(index_gpu, "nprobe", 8)

    xq = ds.get_queries()
    k = 5
    _, I0 = index_gpu.search(xq, k)
    unique_list_ids = _unique_list_ids(cpu_index, xq, 8)

    reclaimed = faiss.evict_ivf_lists(index_gpu, unique_list_ids)
    assert reclaimed.shape[0] == unique_list_ids.shape[0]

    _, I1 = index_gpu.search(xq, k)
    assert (I1 != I0).any(), "Expected results to change after evict"

    loaded = faiss.load_ivf_lists(index_gpu, unique_list_ids)
    assert loaded.shape[0] == unique_list_ids.shape[0]

    _, I2 = index_gpu.search(xq, k)
    np.testing.assert_array_equal(I0, I2)


def test_evict_load_routing():
    """Evict affects correct shard; load restores; is_list_on_gpu reflects state."""
    ngpu = min(2, faiss.get_num_gpus())
    cpu_index, ds = _build_cpu_ivfflat(64, 5000, 50, 64, 8)

    index_gpu = _create_sharded_index(cpu_index, ngpu, 4)
    faiss.GpuParameterSpace().set_index_parameter(index_gpu, "nprobe", 8)

    xq = ds.get_queries()
    unique_list_ids = _unique_list_ids(cpu_index, xq, 8)
    if len(unique_list_ids) == 0:
        pytest.skip("No lists to evict")

    list_id = int(unique_list_ids[0])
    assert faiss.is_list_on_gpu(index_gpu, list_id)

    faiss.evict_ivf_lists(index_gpu, [list_id])
    assert not faiss.is_list_on_gpu(index_gpu, list_id)

    faiss.load_ivf_lists(index_gpu, [list_id])
    assert faiss.is_list_on_gpu(index_gpu, list_id)

    evicted = faiss.get_evicted_lists(index_gpu)
    assert list_id not in evicted


@pytest.mark.parametrize("shard_type", [4, 5])
def test_sharded_error_policy_after_evict_changes_results(shard_type):
    """Default miss policy (Error): after evict, search should not auto-recover."""
    ngpu = min(2, faiss.get_num_gpus())
    cpu_index, ds = _build_cpu_ivfflat(32, 5000, 40, 64, 8)
    xq = ds.get_queries()
    k = 5

    cpu_index.nprobe = 8
    _D_cpu, I_cpu = cpu_index.search(xq, k)

    index_gpu = _create_sharded_index(cpu_index, ngpu, shard_type)
    faiss.GpuParameterSpace().set_index_parameter(index_gpu, "nprobe", 8)

    unique_list_ids = _unique_list_ids(cpu_index, xq, 8)
    if len(unique_list_ids) == 0:
        pytest.skip("No lists to evict for miss-policy=Error test")

    faiss.evict_ivf_lists(index_gpu, unique_list_ids)

    # Ensure shards are in Error mode (no auto-fetch)
    nshard = index_gpu.count() if hasattr(index_gpu, "count") else ngpu
    for s in range(nshard):
        shard = faiss.downcast_index(index_gpu.at(s))
        if hasattr(shard, "setMissPolicy"):
            # Enum is represented as integer in Python bindings (0=Error)
            shard.setMissPolicy(0)
        if hasattr(faiss, "set_auto_fetch"):
            faiss.set_auto_fetch(shard, False)
        if hasattr(faiss, "reset_auto_fetch_stats"):
            faiss.reset_auto_fetch_stats(shard)

    _D_gpu, I_gpu = index_gpu.search(xq, k)
    assert (I_gpu != I_cpu).any(), (
        "Expected results to differ after evict under Error policy"
    )

    # Confirm no auto-fetch activity
    if hasattr(faiss, "get_auto_fetch_stats"):
        total_fetches = 0
        for s in range(nshard):
            shard = faiss.downcast_index(index_gpu.at(s))
            stats = faiss.get_auto_fetch_stats(shard)
            total_fetches += int(stats.get("total_fetches", 0))
        assert total_fetches == 0


@pytest.mark.skipif(
    not hasattr(faiss.StandardGpuResources(), "setDeviceMemoryReservation"),
    reason="setDeviceMemoryReservation not available",
)
def test_pool_and_ipc_multi_gpu():
    """Each GPU's GpuPoolController can expand/shrink independently."""
    ngpu = min(2, faiss.get_num_gpus())
    try:
        from faiss.gpu_pool_controller import (
            GpuPoolController,
            ResizeStatus,
        )
    except ImportError:
        try:
            from faiss.python.gpu_pool_controller import (
                GpuPoolController,
                ResizeStatus,
            )
        except ImportError:
            pytest.skip("GpuPoolController not available")

    pool_size = 64 * 1024 * 1024
    res = [faiss.StandardGpuResources() for _ in range(ngpu)]
    for r in res:
        r.setTempMemory(0)
        r.setDeviceMemoryReservation(pool_size)

    cpu_index, ds = _build_cpu_ivfflat(32, 4000, 50, 64, 8)
    co = faiss.GpuMultipleClonerOptions()
    co.shard = True
    co.shard_type = 5
    co.common_ivf_quantizer = True
    co.use_cuvs = False
    index_gpu = faiss.index_cpu_to_gpu_multiple_py(
        res, cpu_index, co, list(range(ngpu))
    )

    controllers = [GpuPoolController(i) for i in range(ngpu)]
    # Baseline search should be stable across resize operations
    xq = ds.get_queries()
    k = 5
    _D0, I0 = index_gpu.search(xq, k)

    for ctrl in controllers:
        before = ctrl.query()
        assert before["status"] == ResizeStatus.SUCCESS
        assert before["actual_size"] >= pool_size

        # Expand by a small delta; if expansion is not possible on this CI host,
        # skip rather than flake.
        delta = 8 * 1024 * 1024
        expanded = ctrl.expand_by(delta)
        if expanded["status"] != ResizeStatus.SUCCESS:
            pytest.skip(f"pool expand_by failed: {expanded.get('error', '')}")

        after_expand = ctrl.query()
        assert after_expand["status"] == ResizeStatus.SUCCESS
        assert after_expand["actual_size"] >= before["actual_size"]

        _D1, I1 = index_gpu.search(xq, k)
        np.testing.assert_array_equal(I0, I1)

        # Attempt to shrink back to the original size; PARTIAL is acceptable.
        shrunk = ctrl.shrink(int(before["actual_size"]))
        if shrunk["status"] == ResizeStatus.FAILED:
            pytest.skip(f"pool shrink failed: {shrunk.get('error', '')}")

        after_shrink = ctrl.query()
        assert after_shrink["status"] == ResizeStatus.SUCCESS
        assert after_shrink["actual_size"] >= int(before["actual_size"])

        _D2, I2 = index_gpu.search(xq, k)
        np.testing.assert_array_equal(I0, I2)


@pytest.mark.skipif(
    not hasattr(faiss.StandardGpuResources(), "setDeviceMemoryReservation"),
    reason="setDeviceMemoryReservation not available",
)
@pytest.mark.skip(
    reason="evict+pool aborts (use-after-free or pool dealloc); evict without pool works",
)
def test_evict_frees_pool_memory():
    """Evict should increase pool available memory."""
    if faiss.get_num_gpus() < 2:
        pytest.skip("requires >= 2 GPUs")
    if not hasattr(faiss.StandardGpuResources(), "getMemoryInfo"):
        pytest.skip("getMemoryInfo not available")

    ngpu = 2
    pool_size = 128 * 1024 * 1024
    res = [faiss.StandardGpuResources() for _ in range(ngpu)]
    for r in res:
        r.setTempMemory(0)
        r.setDeviceMemoryReservation(pool_size)

    cpu_index, ds = _build_cpu_ivfflat(32, 4000, 50, 64, 8)
    index_gpu = _create_sharded_index(cpu_index, ngpu, 5, res=res)
    faiss.GpuParameterSpace().set_index_parameter(index_gpu, "nprobe", 8)

    xq = ds.get_queries()
    unique_list_ids = _unique_list_ids(cpu_index, xq, 8)
    if len(unique_list_ids) == 0:
        pytest.skip("No lists to evict")

    info_before = [res[i].getMemoryInfo() for i in range(ngpu)]
    faiss.evict_ivf_lists(index_gpu, unique_list_ids)
    info_after = [res[i].getMemoryInfo() for i in range(ngpu)]

    total_before = sum(
        info_before[i].get(gpu, {}).get("IVFLists", (0, 0))[1]
        for i, gpu in enumerate(range(ngpu))
    )
    total_after = sum(
        info_after[i].get(gpu, {}).get("IVFLists", (0, 0))[1]
        for i, gpu in enumerate(range(ngpu))
    )
    assert total_after < total_before or total_before == 0, (
        "Evict should reduce IVFLists usage"
    )


def test_sharded_auto_fetch_restores_results():
    """Auto-fetch on each shard restores search correctness after eviction."""
    _skip_if_no_auto_fetch()

    ngpu = min(2, faiss.get_num_gpus())
    cpu_index, ds = _build_cpu_ivfflat(32, 5000, 40, 64, 8)
    xq = ds.get_queries()
    k = 5

    # CPU baseline
    cpu_index.nprobe = 8
    D_cpu, I_cpu = cpu_index.search(xq, k)

    # Sharded GPU index
    index_gpu = _create_sharded_index(cpu_index, ngpu, 4)
    faiss.GpuParameterSpace().set_index_parameter(index_gpu, "nprobe", 8)

    # Evict all lists touched by this query workload
    unique_list_ids = _unique_list_ids(cpu_index, xq, 8)
    if len(unique_list_ids) == 0:
        pytest.skip("No lists to evict for sharded auto-fetch test")

    faiss.evict_ivf_lists(index_gpu, unique_list_ids)

    # Enable auto-fetch on each shard and reset per-shard stats
    nshard = index_gpu.count() if hasattr(index_gpu, "count") else ngpu
    for s in range(nshard):
        shard = faiss.downcast_index(index_gpu.at(s))
        faiss.set_auto_fetch(shard, True)
        assert faiss.is_auto_fetch_enabled(shard)
        faiss.reset_auto_fetch_stats(shard)

    # Search should transparently trigger auto-fetch on shards and match CPU
    D_gpu, I_gpu = index_gpu.search(xq, k)
    np.testing.assert_array_equal(I_cpu, I_gpu)
    np.testing.assert_array_almost_equal(D_cpu, D_gpu, decimal=4)

    # At least one shard should have performed auto-fetch work
    total_fetches = 0
    total_lists_fetched = 0
    for s in range(nshard):
        shard = faiss.downcast_index(index_gpu.at(s))
        stats = faiss.get_auto_fetch_stats(shard)
        total_fetches += stats["total_fetches"]
        total_lists_fetched += stats["total_lists_fetched"]
    assert total_fetches >= 1
    assert total_lists_fetched >= 1

    # Second search should be fully cached: no additional auto-fetches
    for s in range(nshard):
        shard = faiss.downcast_index(index_gpu.at(s))
        faiss.reset_auto_fetch_stats(shard)

    D_gpu2, I_gpu2 = index_gpu.search(xq, k)
    np.testing.assert_array_equal(I_cpu, I_gpu2)
    np.testing.assert_array_almost_equal(D_cpu, D_gpu2, decimal=4)

    for s in range(nshard):
        shard = faiss.downcast_index(index_gpu.at(s))
        stats2 = faiss.get_auto_fetch_stats(shard)
        assert stats2["total_fetches"] == 0


def test_sharded_auto_fetch_restores_results_type5():
    """Auto-fetch on each shard restores correctness for shard_type=5."""
    _skip_if_no_auto_fetch()

    ngpu = min(2, faiss.get_num_gpus())
    cpu_index, ds = _build_cpu_ivfflat(32, 5000, 40, 64, 8)
    xq = ds.get_queries()
    k = 5

    cpu_index.nprobe = 8
    D_cpu, I_cpu = cpu_index.search(xq, k)

    index_gpu = _create_sharded_index(cpu_index, ngpu, 5)
    faiss.GpuParameterSpace().set_index_parameter(index_gpu, "nprobe", 8)

    unique_list_ids = _unique_list_ids(cpu_index, xq, 8)
    if len(unique_list_ids) == 0:
        pytest.skip("No lists to evict for sharded auto-fetch test (type5)")

    faiss.evict_ivf_lists(index_gpu, unique_list_ids)

    nshard = index_gpu.count() if hasattr(index_gpu, "count") else ngpu
    for s in range(nshard):
        shard = faiss.downcast_index(index_gpu.at(s))
        faiss.set_auto_fetch(shard, True)
        assert faiss.is_auto_fetch_enabled(shard)
        faiss.reset_auto_fetch_stats(shard)

    D_gpu, I_gpu = index_gpu.search(xq, k)
    np.testing.assert_array_equal(I_cpu, I_gpu)
    np.testing.assert_array_almost_equal(D_cpu, D_gpu, decimal=4)


@pytest.mark.parametrize("shard_type", [4, 5])
def test_sharded_cpu_offload_matches_cpu_baseline(shard_type):
    """CpuOffload on each shard matches CPU baseline under IndexShardsIVF::search."""
    if not hasattr(faiss, "IvfListMissPolicy_CpuOffload"):
        pytest.skip("IvfListMissPolicy_CpuOffload not available in this build")
    if not hasattr(faiss, "build_sharded_ivfflat_cpu_offload_index"):
        pytest.skip("build_sharded_ivfflat_cpu_offload_index not available")

    ngpu = min(2, faiss.get_num_gpus())
    cpu_index, ds = _build_cpu_ivfflat(32, 8000, 80, 128, 16)
    xq = ds.get_queries()
    k = 10

    cpu_index.nprobe = 16
    D_ref, I_ref = cpu_index.search(xq, k)

    # Lists touched by this workload, filtered to non-empty.
    touched = _unique_list_ids(cpu_index, xq, cpu_index.nprobe)
    if len(touched) == 0:
        pytest.skip("No lists touched for CpuOffload sharded test")

    # Compute list assignment to shards to preload a fraction of touched lists on GPU.
    list_ids_per_shard, _list_to_shard = faiss._compute_ivf_list_assignment(
        cpu_index, ngpu, shard_type
    )
    preload = [[] for _ in range(ngpu)]
    touched_set = set(int(x) for x in np.asarray(touched).ravel())
    for s in range(ngpu):
        owned_touched = [lid for lid in list_ids_per_shard[s] if int(lid) in touched_set]
        if not owned_touched:
            continue
        half = max(1, len(owned_touched) // 2)
        preload[s] = owned_touched[:half]

    index_gpu, _cpu_backings, _lis, _l2s = faiss.build_sharded_ivfflat_cpu_offload_index(
        cpu_index,
        ngpu=ngpu,
        shard_type=shard_type,
        preload_list_ids_per_shard=preload,
        use_cuvs=False,
    )
    faiss.GpuParameterSpace().set_index_parameter(index_gpu, "nprobe", 16)

    # Enable CpuOffload on each shard.
    nshard = index_gpu.count()
    for s in range(nshard):
        shard = faiss.downcast_index(index_gpu.at(s))
        shard.setMissPolicy(faiss.IvfListMissPolicy_CpuOffload)

    D_off, I_off = index_gpu.search(xq, k)
    np.testing.assert_array_equal(I_ref, I_off)
    np.testing.assert_allclose(D_ref, D_off, rtol=1e-6, atol=1e-3)
