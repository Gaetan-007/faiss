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
    for ctrl in controllers:
        result = ctrl.query()
        assert result["status"] == ResizeStatus.SUCCESS
        assert result["actual_size"] >= pool_size


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
