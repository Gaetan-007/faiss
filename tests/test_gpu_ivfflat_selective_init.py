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
