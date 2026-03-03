import numpy as np
import pytest
import faiss


def _skip_if_no_gpu():
    if not hasattr(faiss, "get_num_gpus") or faiss.get_num_gpus() == 0:
        pytest.skip("GPU not available in this environment")
    required = [
        "GpuIndexIVFFlat",
        "StandardGpuResources",
        "init_ivf_lists_from_cpu",
        "is_list_on_gpu",
        "get_evicted_lists",
        "load_ivf_lists",
        "set_auto_fetch",
    ]
    for name in required:
        if not hasattr(faiss, name):
            pytest.skip(f"Missing faiss GPU symbol: {name}")


def _build_cpu_ivfflat(d=32, nlist=20, nb=2000, seed=123):
    rng = np.random.RandomState(seed)
    xb = rng.rand(nb, d).astype("float32")
    xt = xb[: max(200, nlist * 20)]
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    index.train(xt)
    index.add(xb)
    return index, xb


def _get_nonempty_lists(index):
    il = index.invlists
    return [i for i in range(il.nlist) if il.list_size(i) > 0]


def test_selective_init_populates_cpu_cache():
    _skip_if_no_gpu()
    cpu_index, _ = _build_cpu_ivfflat()
    nonempty = _get_nonempty_lists(cpu_index)
    if len(nonempty) < 2:
        pytest.skip("Need at least two non-empty IVF lists")

    load_ids = nonempty[: max(1, len(nonempty) // 2)]
    evict_ids = [i for i in nonempty if i not in load_ids]

    res = faiss.StandardGpuResources()
    gpu_index = faiss.GpuIndexIVFFlat(
        res, cpu_index.d, cpu_index.nlist, cpu_index.metric_type
    )
    faiss.init_ivf_lists_from_cpu(gpu_index, cpu_index, load_ids)

    for list_id in load_ids:
        assert faiss.is_list_on_gpu(gpu_index, list_id)
    for list_id in evict_ids:
        assert not faiss.is_list_on_gpu(gpu_index, list_id)

    evicted = set(faiss.get_evicted_lists(gpu_index).tolist())
    assert set(evict_ids).issubset(evicted)


def test_load_and_auto_fetch_after_selective_init():
    _skip_if_no_gpu()
    cpu_index, xb = _build_cpu_ivfflat(seed=456)
    nonempty = _get_nonempty_lists(cpu_index)
    if len(nonempty) < 2:
        pytest.skip("Need at least two non-empty IVF lists")

    load_ids = nonempty[:1]
    evict_ids = [i for i in nonempty if i not in load_ids]

    res = faiss.StandardGpuResources()
    gpu_index = faiss.GpuIndexIVFFlat(
        res, cpu_index.d, cpu_index.nlist, cpu_index.metric_type
    )
    faiss.init_ivf_lists_from_cpu(gpu_index, cpu_index, load_ids)

    # Explicitly load one evicted list
    to_load = evict_ids[:1]
    loaded_bytes = faiss.load_ivf_lists(gpu_index, to_load)
    assert int(loaded_bytes[0]) > 0
    assert faiss.is_list_on_gpu(gpu_index, to_load[0])

    # Auto-fetch: request all lists during search
    faiss.set_auto_fetch(gpu_index, True)
    gpu_index.nprobe = cpu_index.nlist
    xq = xb[:10]
    gpu_index.search(xq, 5)

    evicted_after = set(faiss.get_evicted_lists(gpu_index).tolist())
    assert len(evicted_after) == 0


def test_set_no_copy_evict_requires_backing():
    """set_no_copy_evict should fail if no external CPU backing is configured."""
    _skip_if_no_gpu()

    d = 16
    nlist = 10
    res = faiss.StandardGpuResources()
    gpu_index = faiss.GpuIndexIVFFlat(res, d, nlist, faiss.METRIC_L2)

    with pytest.raises(Exception):
        faiss.set_no_copy_evict(gpu_index, True)


def test_no_copy_evict_with_external_backing():
    """No-copy eviction works when the GPU index has external CPU backing."""
    _skip_if_no_gpu()
    cpu_index, _ = _build_cpu_ivfflat()
    nonempty = _get_nonempty_lists(cpu_index)
    if len(nonempty) == 0:
        pytest.skip("Need at least one non-empty IVF list")

    load_ids = nonempty[:1]

    res = faiss.StandardGpuResources()
    gpu_index = faiss.GpuIndexIVFFlat(
        res, cpu_index.d, cpu_index.nlist, cpu_index.metric_type
    )
    faiss.init_ivf_lists_from_cpu(gpu_index, cpu_index, load_ids)

    # Enable no-copy eviction now that we have a CPU backing index.
    faiss.set_no_copy_evict(gpu_index, True)
    list_id = int(load_ids[0])
    assert faiss.is_list_on_gpu(gpu_index, list_id)

    # Evict using the helper; this should route to the no-copy path and
    # rely on external CPU backing instead of a GPU->CPU copy.
    reclaimed = faiss.evict_ivf_lists(gpu_index, [list_id])
    assert reclaimed.shape[0] == 1
    assert not faiss.is_list_on_gpu(gpu_index, list_id)

    evicted = set(faiss.get_evicted_lists(gpu_index).tolist())
    assert list_id in evicted

    # Load the list back from CPU backing and ensure it is resident again.
    loaded = faiss.load_ivf_lists(gpu_index, [list_id])
    assert loaded.shape[0] == 1
    assert faiss.is_list_on_gpu(gpu_index, list_id)


def test_no_copy_evict_search_correctness():
    """
    No-copy eviction preserves search correctness when using a CPU-backed IVF
    index and auto-fetch.
    """
    _skip_if_no_gpu()
    cpu_index, xb = _build_cpu_ivfflat(seed=789)
    nonempty = _get_nonempty_lists(cpu_index)
    if len(nonempty) < 2:
        pytest.skip("Need at least two non-empty IVF lists")

    # Use half of the non-empty lists as initially loaded on GPU
    load_ids = nonempty[: max(1, len(nonempty) // 2)]
    res = faiss.StandardGpuResources()
    gpu_index = faiss.GpuIndexIVFFlat(
        res, cpu_index.d, cpu_index.nlist, cpu_index.metric_type
    )

    # Initialize GPU index with a subset of IVF lists; the remaining lists
    # are backed only by the CPU IndexIVFFlat.
    faiss.init_ivf_lists_from_cpu(gpu_index, cpu_index, load_ids)

    # Enable auto-fetch and no-copy eviction. Auto-fetch will load any
    # evicted/CPU-backed lists that are needed by a search.
    faiss.set_auto_fetch(gpu_index, True)
    faiss.set_no_copy_evict(gpu_index, True)

    # Use a relatively large nprobe to ensure that most lists are visited.
    k = 5
    xq = xb[:20]
    cpu_index.nprobe = cpu_index.nlist
    gpu_index.nprobe = cpu_index.nlist

    # CPU reference result
    D_cpu, I_cpu = cpu_index.search(xq, k)

    # Evict all currently loaded lists via the no-copy path. These lists will
    # subsequently be reloaded from the external CPU backing during search.
    reclaimed = faiss.evict_ivf_lists(gpu_index, load_ids)
    assert reclaimed.shape[0] == len(load_ids)

    # After no-copy eviction, search results should still match the CPU index.
    D_gpu, I_gpu = gpu_index.search(xq, k)
    np.testing.assert_array_equal(I_cpu, I_gpu)
    np.testing.assert_array_almost_equal(D_cpu, D_gpu, decimal=4)
