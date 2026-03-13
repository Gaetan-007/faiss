import os
from pathlib import Path

import numpy as np
import faiss
import pytest
import torch

from torch.cuda import nvtx


pytestmark = pytest.mark.skipif(
    faiss.get_num_gpus() < 1,
    reason="gpu only test",
)


def _build_cpu_ivfflat_index(xb, d, nlist, metric, nprobe):
    quantizer = faiss.IndexFlatL2(d)
    cpu_index = faiss.IndexIVFFlat(quantizer, d, nlist, metric)
    cpu_index.train(xb)
    cpu_index.add(xb)
    cpu_index.nprobe = nprobe
    return cpu_index


def _to_gpu(cpu_index, device):
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, device, cpu_index)
    gpu_index.nprobe = cpu_index.nprobe
    return gpu_index


def _unique_list_ids(cpu_index, xq, nprobe):
    _, list_ids = cpu_index.quantizer.search(xq, nprobe)
    return np.unique(list_ids.reshape(-1))


def test_manual_evict_load():
    d = 1024
    nb = 10000
    nq = 20
    nlist = 100
    k = 5

    rng = np.random.RandomState(123)
    xb = rng.random((nb, d)).astype("float32")
    xq = xb[:nq].copy()

    cpu_index = _build_cpu_ivfflat_index(
        xb=xb,
        d=d,
        nlist=nlist,
        metric=faiss.METRIC_L2,
        nprobe=1,
    )

    nvtx.range_push("index_cpu_to_gpu")
    torch.cuda.synchronize()
    gpu_index = _to_gpu(cpu_index, 0)
    nvtx.range_pop()

    nvtx.range_push("search_baseline")
    _, I0 = gpu_index.search(xq, k)
    nvtx.range_pop()

    unique_list_ids = _unique_list_ids(cpu_index, xq, 1)

    nvtx.range_push("evict_ivf_lists")
    reclaimed = faiss.evict_ivf_lists(gpu_index, unique_list_ids)
    nvtx.range_pop()
    assert reclaimed.shape[0] == unique_list_ids.shape[0]

    _, I1 = gpu_index.search(xq, k)
    changed = I1[:, 0] != I0[:, 0]
    assert changed.any(), "Expected some top-1 results to change after eviction"

    nvtx.range_push("load_ivf_lists")
    loaded = faiss.load_ivf_lists(gpu_index, unique_list_ids)
    nvtx.range_pop()
    assert loaded.shape[0] == unique_list_ids.shape[0]

    _, I2 = gpu_index.search(xq, k)
    assert np.array_equal(I0, I2), "Results after reload do not match baseline"


def test_auto_fetch():
    d = 128
    nb = 10000
    nq = 50
    nlist = 100
    k = 10

    rng = np.random.RandomState(456)
    xb = rng.random((nb, d)).astype("float32")
    xq = rng.random((nq, d)).astype("float32")

    cpu_index = _build_cpu_ivfflat_index(
        xb=xb,
        d=d,
        nlist=nlist,
        metric=faiss.METRIC_L2,
        nprobe=10,
    )
    gpu_index = _to_gpu(cpu_index, 0)

    _, I_baseline = gpu_index.search(xq, k)

    assert not faiss.is_auto_fetch_enabled(gpu_index)

    unique_list_ids = _unique_list_ids(cpu_index, xq, gpu_index.nprobe)
    lists_to_evict = unique_list_ids[: len(unique_list_ids) // 2]
    reclaimed = faiss.evict_ivf_lists(gpu_index, lists_to_evict)
    assert reclaimed.shape[0] == lists_to_evict.shape[0]

    evicted_lists = faiss.get_evicted_lists(gpu_index)
    assert len(evicted_lists) == len(lists_to_evict)

    for lid in lists_to_evict:
        assert not faiss.is_list_on_gpu(
            gpu_index, lid
        ), f"List {lid} should not be on GPU after eviction"

    _, I_no_fetch = gpu_index.search(xq, k)
    assert not np.array_equal(
        I_baseline, I_no_fetch
    ), "Expected search results to change without auto-fetch"

    faiss.set_auto_fetch(gpu_index, True)
    assert faiss.is_auto_fetch_enabled(gpu_index)

    faiss.reset_auto_fetch_stats(gpu_index)
    nvtx.range_push("search_with_auto_fetch")
    _, I_auto_fetch = gpu_index.search(xq, k)
    nvtx.range_pop()

    stats = faiss.get_auto_fetch_stats(gpu_index)
    assert stats["total_lists_fetched"] >= 0
    assert stats["total_bytes_fetched"] >= 0

    still_evicted = faiss.get_evicted_lists(gpu_index)
    assert len(still_evicted) <= len(lists_to_evict)

    faiss.reset_auto_fetch_stats(gpu_index)
    _, I_cached = gpu_index.search(xq, k)
    stats2 = faiss.get_auto_fetch_stats(gpu_index)
    assert stats2["total_fetches"] == 0
    assert np.array_equal(
        I_auto_fetch, I_cached
    ), "Results should be identical for consecutive searches"

    faiss.set_auto_fetch(gpu_index, False)
    assert not faiss.is_auto_fetch_enabled(gpu_index)


def test_auto_fetch_with_all_lists_evicted():
    d = 64
    nb = 5000
    nq = 10
    nlist = 50
    k = 5

    rng = np.random.RandomState(789)
    xb = rng.random((nb, d)).astype("float32")
    xq = rng.random((nq, d)).astype("float32")

    cpu_index = _build_cpu_ivfflat_index(
        xb=xb,
        d=d,
        nlist=nlist,
        metric=faiss.METRIC_L2,
        nprobe=5,
    )
    gpu_index = _to_gpu(cpu_index, 0)

    _, I_baseline = gpu_index.search(xq, k)

    unique_list_ids = _unique_list_ids(cpu_index, xq, gpu_index.nprobe)
    faiss.evict_ivf_lists(gpu_index, unique_list_ids)

    faiss.set_auto_fetch(gpu_index, True)
    faiss.reset_auto_fetch_stats(gpu_index)

    _, I_result = gpu_index.search(xq, k)
    stats = faiss.get_auto_fetch_stats(gpu_index)
    assert stats["total_lists_fetched"] >= len(unique_list_ids)
    assert np.array_equal(I_baseline, I_result), "Results should match baseline"

    faiss.set_auto_fetch(gpu_index, False)


def test_auto_fetch_miss_logging(tmp_path, monkeypatch):
    d = 64
    nb = 5000
    nq = 20
    nlist = 50
    k = 5

    # Isolate log output under a temporary HOME directory
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    rng = np.random.RandomState(321)
    xb = rng.random((nb, d)).astype("float32")
    xq = rng.random((nq, d)).astype("float32")

    cpu_index = _build_cpu_ivfflat_index(
        xb=xb,
        d=d,
        nlist=nlist,
        metric=faiss.METRIC_L2,
        nprobe=5,
    )
    gpu_index = _to_gpu(cpu_index, 0)

    # Ensure there are IVF lists touched by this workload.
    unique_list_ids = _unique_list_ids(cpu_index, xq, gpu_index.nprobe)
    if unique_list_ids.size == 0:
        pytest.skip("No IVF lists are used by this query workload")

    # Evict all probed lists so that subsequent search under AutoFetch policy
    # will need to load them back from CPU and be logged as misses.
    faiss.evict_ivf_lists(gpu_index, unique_list_ids)

    faiss.set_auto_fetch(gpu_index, True)
    assert faiss.is_auto_fetch_enabled(gpu_index)

    # Trigger a search; this should cause AutoFetch to load missing lists and
    # emit miss logs for the queries whose probed lists are evicted.
    gpu_index.search(xq, k)

    log_dir = fake_home / ".faiss_log"
    assert log_dir.is_dir(), "miss log directory was not created"

    log_files = sorted(log_dir.glob("miss_log_*.log"))
    assert log_files, "no miss log file created for AutoFetch policy"

    latest = log_files[-1]
    lines = latest.read_text(encoding="utf-8").strip().splitlines()
    assert lines, "miss log file is empty"

    last = lines[-1]
    parts = last.split(",")
    assert len(parts) == 4, f"unexpected log line format: {last!r}"

    # Format: timestamp, miss_count, miss_centroids, policy
    timestamp, miss_count_str, miss_centroids_str, policy_str = parts
    assert policy_str == "AutoFetch"
    miss_count = int(miss_count_str)
    assert miss_count > 0
    assert miss_centroids_str, "miss centroid list should not be empty"
