#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Demo of copy-based IVF list eviction on GPU (default behavior).

This script shows how to:

- Build a CPU IndexIVFFlat as the reference index.
- Clone it to a single-GPU GpuIndexIVFFlat.
- Evict IVF lists from GPU using the default (copy-based) eviction:
  the list data is copied back to host and stored in cpuListCache_.
- Observe that search results change after eviction when auto-fetch is
  disabled.
- Load the lists back to GPU and verify that results again match CPU.

Run with:

    python3 demos/demo_copy_evict_ivfflat.py
"""

import sys
import numpy as np

import faiss


def _require(condition, msg):
    if not condition:
        raise RuntimeError(msg)


def _check_gpu_and_symbols():
    if not hasattr(faiss, "get_num_gpus") or faiss.get_num_gpus() == 0:
        raise RuntimeError("GPU not available in this environment")

    required = [
        "StandardGpuResources",
        "index_cpu_to_gpu",
        "evict_ivf_lists",
        "load_ivf_lists",
        "is_list_on_gpu",
        "get_evicted_lists",
    ]
    missing = [name for name in required if not hasattr(faiss, name)]
    if missing:
        raise RuntimeError(
            "Missing required faiss GPU symbols for this demo: "
            + ", ".join(missing)
        )


def _build_cpu_ivfflat(d=64, nlist=128, nb=20000, seed=123):
    rng = np.random.RandomState(seed)
    xb = rng.rand(nb, d).astype("float32")
    xt = xb[: max(200, nlist * 20)]

    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    index.train(xt)
    index.add(xb)
    return index, xb


def _choose_nonempty_lists(cpu_index, xq, nprobe=8):
    # Use the coarse quantizer on CPU to see which IVF lists are visited
    _, list_ids = cpu_index.quantizer.search(xq, nprobe)
    ids = np.unique(list_ids.reshape(-1))
    non_empty = [
        int(i)
        for i in ids
        if 0 <= int(i) < cpu_index.nlist
        and cpu_index.invlists.list_size(int(i)) > 0
    ]
    return non_empty


def run_demo():
    _check_gpu_and_symbols()

    cpu_index, xb = _build_cpu_ivfflat()
    print(f"CPU IndexIVFFlat built with {cpu_index.ntotal} vectors, nlist={cpu_index.nlist}.")

    # Clone to GPU: this creates a GpuIndexIVFFlat with all lists resident
    # on the device. No external backing is registered, so eviction will
    # follow the default copy-based behavior.
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

    # Prepare a small query set and find some non-empty IVF lists that will
    # actually be touched during search.
    xq = xb[:200]
    cpu_index.nprobe = 8
    gpu_index.nprobe = cpu_index.nprobe

    candidate_lists = _choose_nonempty_lists(cpu_index, xq, nprobe=cpu_index.nprobe)
    _require(
        len(candidate_lists) > 0,
        "No non-empty IVF lists discovered for eviction; "
        "try increasing nb or adjusting nlist/nprobe.",
    )

    # Use a subset of these lists for eviction.
    evict_ids = candidate_lists[: min(16, len(candidate_lists))]
    print(f"Will evict {len(evict_ids)} IVF lists (copy-based eviction).")

    # Baseline: CPU vs GPU search before any eviction.
    k = 10
    print("Running baseline CPU and GPU searches...")
    D_cpu, I_cpu = cpu_index.search(xq, k)
    D_gpu0, I_gpu0 = gpu_index.search(xq, k)

    np.testing.assert_array_equal(I_cpu, I_gpu0)
    np.testing.assert_array_almost_equal(D_cpu, D_gpu0, decimal=4)
    print("Baseline check passed: GPU results match CPU before eviction.")

    # Sanity-check that these lists are actually on GPU.
    for list_id in evict_ids:
        _require(
            faiss.is_list_on_gpu(gpu_index, int(list_id)),
            f"List {list_id} was expected to be on GPU before eviction",
        )

    # Disable auto-fetch so the effect of eviction is visible in results.
    if hasattr(faiss, "set_auto_fetch"):
        faiss.set_auto_fetch(gpu_index, False)

    print("Evicting selected IVF lists (copy-based eviction)...")
    reclaimed = faiss.evict_ivf_lists(gpu_index, evict_ids)
    print("Reclaimed bytes per list:")
    print(reclaimed)

    # After copy-based eviction, these lists should no longer be on GPU, but
    # their contents are stored in the GPU index's cpuListCache_.
    for list_id in evict_ids:
        _require(
            not faiss.is_list_on_gpu(gpu_index, int(list_id)),
            f"List {list_id} should have been evicted from GPU",
        )

    evicted = set(faiss.get_evicted_lists(gpu_index).tolist())
    print(f"Number of lists currently marked as evicted: {len(evicted)}")

    # Search again: since auto-fetch is disabled, results should generally
    # differ from the CPU reference (we are missing part of the data on GPU).
    print("Running GPU search after eviction (auto-fetch disabled)...")
    D_gpu1, I_gpu1 = gpu_index.search(xq, k)

    if np.array_equal(I_cpu, I_gpu1):
        print(
            "WARNING: GPU results still match CPU after eviction; this can "
            "happen if the evicted lists were not needed for these queries."
        )
    else:
        print("As expected, some results changed after eviction.")

    # Now load the evicted lists back from the copy stored in cpuListCache_.
    print("Loading evicted IVF lists back to GPU...")
    loaded = faiss.load_ivf_lists(gpu_index, evict_ids)
    print("Loaded bytes per list:")
    print(loaded)

    for list_id in evict_ids:
        _require(
            faiss.is_list_on_gpu(gpu_index, int(list_id)),
            f"List {list_id} should be back on GPU after load",
        )

    evicted_after = set(faiss.get_evicted_lists(gpu_index).tolist())
    print(f"Number of lists still marked as evicted: {len(evicted_after)}")

    # Final search: results should again line up with the CPU reference.
    print("Running GPU search after re-loading lists...")
    D_gpu2, I_gpu2 = gpu_index.search(xq, k)

    np.testing.assert_array_equal(I_cpu, I_gpu2)
    np.testing.assert_array_almost_equal(D_cpu, D_gpu2, decimal=4)
    print("Final check passed: GPU results match CPU after load.")

    print("\nCopy-based eviction demo complete.")
    print(
        "You can change 'd', 'nb', 'nlist', or which lists are evicted to "
        "explore different behaviors."
    )


if __name__ == "__main__":
    try:
        run_demo()
    except Exception as e:
        print(f"[demo_copy_evict_ivfflat] ERROR: {e}", file=sys.stderr)
        sys.exit(1)

