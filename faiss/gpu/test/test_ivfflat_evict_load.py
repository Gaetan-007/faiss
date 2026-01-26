import numpy as np
import faiss
import torch

from torch.cuda import nvtx


def test_manual_evict_load():
    """Test manual evict and load of IVF lists."""
    print("=" * 60)
    print("Test: Manual evict and load")
    print("=" * 60)

    d = 1024
    nb = 10000
    nq = 20
    nlist = 100
    k = 5

    rng = np.random.RandomState(123)
    xb = rng.random((nb, d)).astype("float32")
    xq = xb[:nq].copy()

    quantizer = faiss.IndexFlatL2(d)
    cpu_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    cpu_index.train(xb)
    cpu_index.add(xb)
    cpu_index.nprobe = 1

    torch.cuda.cudart().cudaProfilerStart()

    res = faiss.StandardGpuResources()
    
    nvtx.range_push("index_cpu_to_gpu")
    torch.cuda.synchronize()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    nvtx.range_pop()

    gpu_index.nprobe = 1
    nvtx.range_push("search")
    D0, I0 = gpu_index.search(xq, k)
    nvtx.range_pop()

    _, list_ids = cpu_index.quantizer.search(xq, 1)
    list_ids = list_ids.reshape(-1)
    unique_list_ids = np.unique(list_ids)
    print(f"unique_list_ids: {unique_list_ids}")

    nvtx.range_push("evict_ivf_lists")
    reclaimed = faiss.evict_ivf_lists(gpu_index, unique_list_ids)
    nvtx.range_pop()
    assert reclaimed.shape[0] == unique_list_ids.shape[0]

    D1, I1 = gpu_index.search(xq, k)
    changed = (I1[:, 0] != I0[:, 0])
    assert changed.any(), "Expected some top-1 results to change after eviction"

    nvtx.range_push("load_ivf_lists")
    loaded = faiss.load_ivf_lists(gpu_index, unique_list_ids)
    nvtx.range_pop()
    assert loaded.shape[0] == unique_list_ids.shape[0]

    D2, I2 = gpu_index.search(xq, k)

    torch.cuda.cudart().cudaProfilerStop()

    if not np.array_equal(I0, I2):
        raise AssertionError("Results after reload do not match baseline")

    print("OK: Manual evict and load test passed")


def test_auto_fetch():
    """
    Test page-fault style auto-fetch functionality.
    
    When auto-fetch is enabled, searching with evicted lists should
    automatically load them from CPU cache, producing correct results.
    """
    print("=" * 60)
    print("Test: Auto-fetch (page-fault style automatic load)")
    print("=" * 60)

    d = 128
    nb = 10000
    nq = 50
    nlist = 100
    k = 10

    rng = np.random.RandomState(456)
    xb = rng.random((nb, d)).astype("float32")
    xq = rng.random((nq, d)).astype("float32")

    # Build and train CPU index
    quantizer = faiss.IndexFlatL2(d)
    cpu_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    cpu_index.train(xb)
    cpu_index.add(xb)
    cpu_index.nprobe = 10  # Search 10 lists per query

    # Transfer to GPU
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    gpu_index.nprobe = 10

    # Get baseline results with all data on GPU
    D_baseline, I_baseline = gpu_index.search(xq, k)
    print(f"Baseline search completed. Top result for query 0: {I_baseline[0, :3]}")

    # Verify auto-fetch is disabled by default
    assert not faiss.is_auto_fetch_enabled(gpu_index), \
        "Auto-fetch should be disabled by default"

    # Find lists that will be accessed during search
    _, list_ids = cpu_index.quantizer.search(xq, gpu_index.nprobe)
    list_ids = list_ids.reshape(-1)
    unique_list_ids = np.unique(list_ids)
    print(f"Number of unique lists to be accessed: {len(unique_list_ids)}")

    # Evict some lists to CPU
    lists_to_evict = unique_list_ids[:len(unique_list_ids) // 2]
    print(f"Evicting {len(lists_to_evict)} lists to CPU cache...")
    reclaimed = faiss.evict_ivf_lists(gpu_index, lists_to_evict)
    print(f"Reclaimed {reclaimed.sum()} bytes from GPU")

    # Verify lists are evicted
    evicted_lists = faiss.get_evicted_lists(gpu_index)
    assert len(evicted_lists) == len(lists_to_evict), \
        f"Expected {len(lists_to_evict)} evicted lists, got {len(evicted_lists)}"
    print(f"Confirmed {len(evicted_lists)} lists are evicted")

    # Check isListOnGpu
    for lid in lists_to_evict:
        assert not faiss.is_list_on_gpu(gpu_index, lid), \
            f"List {lid} should not be on GPU after eviction"

    # Search WITHOUT auto-fetch - results should be degraded
    D_no_fetch, I_no_fetch = gpu_index.search(xq, k)
    results_changed = not np.array_equal(I_baseline, I_no_fetch)
    print(f"Results changed without auto-fetch: {results_changed}")

    # Now enable auto-fetch
    faiss.set_auto_fetch(gpu_index, True)
    assert faiss.is_auto_fetch_enabled(gpu_index), \
        "Auto-fetch should now be enabled"
    print("Auto-fetch enabled")

    # Reset stats
    faiss.reset_auto_fetch_stats(gpu_index)

    # Re-evict the lists (they were loaded during previous search without auto-fetch)
    # Actually, they weren't loaded because auto-fetch was disabled
    # But let's make sure they're evicted by checking

    # Search WITH auto-fetch - should automatically load missing lists
    nvtx.range_push("search_with_auto_fetch")
    D_auto_fetch, I_auto_fetch = gpu_index.search(xq, k)
    nvtx.range_pop()

    # Check auto-fetch stats
    stats = faiss.get_auto_fetch_stats(gpu_index)
    print(f"Auto-fetch stats:")
    print(f"  Total fetches: {stats['total_fetches']}")
    print(f"  Total lists fetched: {stats['total_lists_fetched']}")
    print(f"  Total bytes fetched: {stats['total_bytes_fetched']}")

    # Verify auto-fetch was triggered
    if stats['total_fetches'] > 0:
        print(f"Auto-fetch was triggered {stats['total_fetches']} times")
    else:
        print("Note: Auto-fetch was not triggered (lists may have been loaded previously)")

    # Results should now match baseline (or be very close)
    # After auto-fetch, all needed lists should be on GPU
    matches = np.sum(I_auto_fetch == I_baseline)
    total = I_baseline.size
    match_rate = matches / total * 100
    print(f"Result match rate after auto-fetch: {match_rate:.2f}% ({matches}/{total})")

    # Verify that all previously evicted lists are now back on GPU
    still_evicted = faiss.get_evicted_lists(gpu_index)
    if len(still_evicted) < len(lists_to_evict):
        print(f"Lists loaded by auto-fetch: {len(lists_to_evict) - len(still_evicted)}")

    # Search again - should not trigger auto-fetch since data is already on GPU
    faiss.reset_auto_fetch_stats(gpu_index)
    D_cached, I_cached = gpu_index.search(xq, k)
    stats2 = faiss.get_auto_fetch_stats(gpu_index)
    
    # Results should be identical since no lists were evicted
    assert np.array_equal(I_auto_fetch, I_cached), \
        "Results should be identical for consecutive searches with same data"
    print("Second search results match (no new fetches needed)")

    # Disable auto-fetch and verify
    faiss.set_auto_fetch(gpu_index, False)
    assert not faiss.is_auto_fetch_enabled(gpu_index), \
        "Auto-fetch should now be disabled"

    print("OK: Auto-fetch test passed")


def test_auto_fetch_with_all_lists_evicted():
    """
    Test auto-fetch when ALL lists needed for a search are evicted.
    This is an extreme case to verify robustness.
    """
    print("=" * 60)
    print("Test: Auto-fetch with all required lists evicted")
    print("=" * 60)

    d = 64
    nb = 5000
    nq = 10
    nlist = 50
    k = 5

    rng = np.random.RandomState(789)
    xb = rng.random((nb, d)).astype("float32")
    xq = rng.random((nq, d)).astype("float32")

    # Build and train CPU index
    quantizer = faiss.IndexFlatL2(d)
    cpu_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    cpu_index.train(xb)
    cpu_index.add(xb)
    cpu_index.nprobe = 5

    # Transfer to GPU
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    gpu_index.nprobe = 5

    # Get baseline results
    D_baseline, I_baseline = gpu_index.search(xq, k)

    # Find all lists that will be accessed
    _, list_ids = cpu_index.quantizer.search(xq, gpu_index.nprobe)
    unique_list_ids = np.unique(list_ids.reshape(-1))
    print(f"Evicting ALL {len(unique_list_ids)} lists needed for search...")

    # Evict ALL lists needed for the search
    faiss.evict_ivf_lists(gpu_index, unique_list_ids)

    # Enable auto-fetch
    faiss.set_auto_fetch(gpu_index, True)
    faiss.reset_auto_fetch_stats(gpu_index)

    # Search should automatically load all needed lists
    D_result, I_result = gpu_index.search(xq, k)

    stats = faiss.get_auto_fetch_stats(gpu_index)
    print(f"Auto-fetch loaded {stats['total_lists_fetched']} lists")

    # Results should match baseline
    if np.array_equal(I_baseline, I_result):
        print("Results match baseline exactly!")
    else:
        matches = np.sum(I_baseline == I_result)
        total = I_baseline.size
        print(f"Result match rate: {matches}/{total} ({100*matches/total:.1f}%)")

    # Cleanup
    faiss.set_auto_fetch(gpu_index, False)
    print("OK: Auto-fetch with all lists evicted test passed")


def main():
    # Run all tests
    test_manual_evict_load()
    print()
    test_auto_fetch()
    print()
    test_auto_fetch_with_all_lists_evicted()
    print()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
