import os
from pathlib import Path

import numpy as np
import pytest

import faiss


pytestmark = pytest.mark.skipif(
    faiss.get_num_gpus() < 1,
    reason="gpu only test",
)


def _build_cpu_ivfflat(xb, d, nlist, metric, nprobe):
    quantizer = faiss.IndexFlatL2(d) if metric == faiss.METRIC_L2 else faiss.IndexFlatIP(d)
    cpu_index = faiss.IndexIVFFlat(quantizer, d, nlist, metric)
    cpu_index.train(xb)
    cpu_index.add(xb)
    cpu_index.nprobe = nprobe
    return cpu_index


def _unique_list_ids(cpu_index, xq, nprobe):
    _, list_ids = cpu_index.quantizer.search(xq, nprobe)
    return np.unique(list_ids.reshape(-1))


def _has_cpu_offload_enum():
    return hasattr(faiss, "IvfListMissPolicy_CpuOffload")


def test_cpu_offload_matches_cpu_baseline():
    if not _has_cpu_offload_enum():
        pytest.skip("IvfListMissPolicy_CpuOffload not available in this build")

    d = 1024
    nb = 1024 * 1024
    nq = 50
    nlist = 1024
    nprobe = 128
    k = 10

    rng = np.random.RandomState(123)
    xb = rng.random((nb, d)).astype("float32")
    xq = rng.random((nq, d)).astype("float32")

    cpu_index = _build_cpu_ivfflat(
        xb=xb,
        d=d,
        nlist=nlist,
        metric=faiss.METRIC_L2,
        nprobe=nprobe,
    )

    # CPU baseline
    D_ref, I_ref = cpu_index.search(xq, k)

    # Determine which lists are touched by this workload
    unique_ids = _unique_list_ids(cpu_index, xq, cpu_index.nprobe)
    if unique_ids.size == 0:
        pytest.skip("No IVF lists are used by this query workload")

    # Load roughly half of the touched lists to GPU; the rest remain CPU-only
    half = max(1, unique_ids.size // 2)
    lists_on_gpu = unique_ids[:half]

    res = faiss.StandardGpuResources()
    gpu_index = faiss.GpuIndexIVFFlat(
        res,
        cpu_index.d,
        cpu_index.nlist,
        cpu_index.metric_type,
    )

    # copyFromSelective initializes the GPU index with selected lists on GPU and
    # all remaining non-empty lists backed by the CPU index.
    faiss.init_ivf_lists_from_cpu(gpu_index, cpu_index, lists_on_gpu)
    gpu_index.nprobe = cpu_index.nprobe

    # Enable CPU offload miss policy
    gpu_index.setMissPolicy(faiss.IvfListMissPolicy_CpuOffload)

    D_off, I_off = gpu_index.search(xq, k)

    np.testing.assert_array_equal(I_ref, I_off)
    # Allow slightly looser absolute tolerance to accommodate mixed CPU+GPU
    # search paths while still catching meaningful numerical regressions.
    np.testing.assert_allclose(D_ref, D_off, rtol=1e-6, atol=1e-3)


def test_cpu_offload_topk_indices_match_cpu_baseline():
    """
    Validate that CpuOffload returns the same top-k neighbor IDs as the pure
    CPU baseline, ignoring absolute distance scores.
    """
    if not _has_cpu_offload_enum():
        pytest.skip("IvfListMissPolicy_CpuOffload not available in this build")

    d = 1024
    nb = 1024 * 1024
    nq = 50
    nlist = 1024
    nprobe = 128
    k = 10

    rng = np.random.RandomState(123)
    xb = rng.random((nb, d)).astype("float32")
    xq = rng.random((nq, d)).astype("float32")

    cpu_index = _build_cpu_ivfflat(
        xb=xb,
        d=d,
        nlist=nlist,
        metric=faiss.METRIC_L2,
        nprobe=nprobe,
    )

    D_ref, I_ref = cpu_index.search(xq, k)

    unique_ids = _unique_list_ids(cpu_index, xq, cpu_index.nprobe)
    if unique_ids.size == 0:
        pytest.skip("No IVF lists are used by this query workload")

    half = max(1, unique_ids.size // 2)
    lists_on_gpu = unique_ids[:half]

    res = faiss.StandardGpuResources()
    gpu_index = faiss.GpuIndexIVFFlat(
        res,
        cpu_index.d,
        cpu_index.nlist,
        cpu_index.metric_type,
    )

    faiss.init_ivf_lists_from_cpu(gpu_index, cpu_index, lists_on_gpu)
    gpu_index.nprobe = cpu_index.nprobe
    gpu_index.setMissPolicy(faiss.IvfListMissPolicy_CpuOffload)

    D_off, I_off = gpu_index.search(xq, k)

    np.testing.assert_array_equal(I_ref, I_off)


def test_cpu_offload_requires_external_backing():
    if not _has_cpu_offload_enum():
        pytest.skip("IvfListMissPolicy_CpuOffload not available in this build")

    d = 1024
    nb = 1024 * 1024
    nq = 10
    nlist = 1024
    nprobe = 128
    k = 5

    rng = np.random.RandomState(456)
    xb = rng.random((nb, d)).astype("float32")
    xq = rng.random((nq, d)).astype("float32")

    # Standard GPU index without external CPU backing
    cpu_index = _build_cpu_ivfflat(
        xb=xb,
        d=d,
        nlist=nlist,
        metric=faiss.METRIC_L2,
        nprobe=nprobe,
    )
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    gpu_index.nprobe = cpu_index.nprobe

    # Manually evict some lists so that the miss-policy logic would need CPU data
    unique_ids = _unique_list_ids(cpu_index, xq, gpu_index.nprobe)
    if unique_ids.size == 0:
        pytest.skip("No IVF lists are used by this query workload")
    evict_ids = unique_ids[: max(1, unique_ids.size // 2)]
    faiss.evict_ivf_lists(gpu_index, evict_ids)

    # Switching to CpuOffload without an external backing index should raise
    # when performing search.
    gpu_index.setMissPolicy(faiss.IvfListMissPolicy_CpuOffload)
    with pytest.raises(Exception):
        gpu_index.search(xq, k)


def test_cpu_offload_miss_logging(tmp_path, monkeypatch):
    if not _has_cpu_offload_enum():
        pytest.skip("IvfListMissPolicy_CpuOffload not available in this build")

    # Isolate log output under a temporary HOME directory
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    d = 64
    nb = 2000
    nq = 10
    nlist = 64
    nprobe = 8
    k = 5

    rng = np.random.RandomState(7)
    xb = rng.random((nb, d)).astype("float32")
    xq = rng.random((nq, d)).astype("float32")

    cpu_index = _build_cpu_ivfflat(
        xb=xb,
        d=d,
        nlist=nlist,
        metric=faiss.METRIC_L2,
        nprobe=nprobe,
    )

    unique_ids = _unique_list_ids(cpu_index, xq, cpu_index.nprobe)
    if unique_ids.size < 2:
        pytest.skip("Not enough IVF lists touched to induce misses")

    # Load roughly half of the touched lists to GPU; the rest remain CPU-backed
    half = max(1, unique_ids.size // 2)
    lists_on_gpu = unique_ids[:half]

    res = faiss.StandardGpuResources()
    gpu_index = faiss.GpuIndexIVFFlat(
        res,
        cpu_index.d,
        cpu_index.nlist,
        cpu_index.metric_type,
    )

    faiss.init_ivf_lists_from_cpu(gpu_index, cpu_index, lists_on_gpu)
    gpu_index.nprobe = cpu_index.nprobe

    # Enable CpuOffload miss policy so that probed lists that were evicted
    # will be searched on CPU and logged as misses.
    gpu_index.setMissPolicy(faiss.IvfListMissPolicy_CpuOffload)

    # Trigger at least one search; this should produce miss logs for queries
    # whose probed lists are backed only by CPU.
    gpu_index.search(xq, k)

    log_dir = fake_home / ".faiss_log"
    assert log_dir.is_dir(), "miss log directory was not created"

    log_files = sorted(log_dir.glob("miss_log_*.log"))
    assert log_files, "no miss log file created for CpuOffload policy"

    # Inspect the latest log file
    latest = log_files[-1]
    lines = latest.read_text(encoding="utf-8").strip().splitlines()
    assert lines, "miss log file is empty"

    last = lines[-1]
    parts = last.split(",")
    assert len(parts) == 4, f"unexpected log line format: {last!r}"

    # Format: timestamp, miss_count, miss_centroids, policy
    timestamp, miss_count_str, miss_centroids_str, policy_str = parts
    assert policy_str == "CpuOffload"
    miss_count = int(miss_count_str)
    assert miss_count > 0
    assert miss_centroids_str, "miss centroid list should not be empty"

