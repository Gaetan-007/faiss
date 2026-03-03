#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Simple demo of no-copy IVF list eviction on GPU.

This script shows how to:

- Build a CPU IndexIVFFlat as the authoritative backing store.
- Initialize a GpuIndexIVFFlat by loading only a subset of IVF lists to GPU
  (the rest remain in CPU memory).
- Enable page-fault style auto-fetch and no-copy eviction.
- Evict IVF lists on GPU without GPU->CPU copies, relying on the CPU index
  for backing storage.
- Verify that GPU search results still match the CPU index after eviction.

Run with:

    python3 demos/demo_no_copy_evict_ivfflat.py
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
        "GpuIndexIVFFlat",
        "StandardGpuResources",
        "init_ivf_lists_from_cpu",
        "evict_ivf_lists",
        "load_ivf_lists",
        "set_auto_fetch",
        "set_no_copy_evict",
        "is_list_on_gpu",
        "get_evicted_lists",
    ]
    missing = [name for name in required if not hasattr(faiss, name)]
    if missing:
        raise RuntimeError(
            "Missing required faiss GPU symbols for this demo: "
            + ", ".join(missing)
        )


def _build_cpu_ivfflat(d=64, nlist=64, nb=20000, seed=123):
    rng = np.random.RandomState(seed)
    xb = rng.rand(nb, d).astype("float32")
    xt = xb[: max(200, nlist * 20)]

    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    index.train(xt)
    index.add(xb)
    return index, xb


def _get_nonempty_lists(index):
    invlists = index.invlists
    return [i for i in range(invlists.nlist) if invlists.list_size(i) > 0]


def run_demo():
    _check_gpu_and_symbols()

    cpu_index, xb = _build_cpu_ivfflat()
    nonempty = _get_nonempty_lists(cpu_index)
    _require(
        len(nonempty) >= 2,
        "Need at least two non-empty IVF lists for the demo; "
        "try increasing nb or nlist.",
    )

    # Load only a subset of IVF lists to GPU initially; the rest will stay
    # in CPU memory as backing storage.
    load_ids = nonempty[: max(1, len(nonempty) // 2)]
    print(f"CPU index built with {cpu_index.ntotal} vectors.")
    print(f"Non-empty IVF lists: {len(nonempty)}; initially loading {len(load_ids)}.")

    res = faiss.StandardGpuResources()
    gpu_index = faiss.GpuIndexIVFFlat(
        res, cpu_index.d, cpu_index.nlist, cpu_index.metric_type
    )

    # Initialize GPU IVF lists selectively from the CPU index.
    faiss.init_ivf_lists_from_cpu(gpu_index, cpu_index, load_ids)

    for list_id in load_ids:
        _require(
            faiss.is_list_on_gpu(gpu_index, int(list_id)),
            f"List {list_id} was expected on GPU after init_ivf_lists_from_cpu",
        )

    evicted_initial = set(faiss.get_evicted_lists(gpu_index).tolist())
    print(f"Evicted lists right after init: {len(evicted_initial)}")

    # Enable page-fault style auto-fetch and no-copy eviction.
    faiss.set_auto_fetch(gpu_index, True)
    faiss.set_no_copy_evict(gpu_index, True)

    # Use a reasonably large nprobe so that we touch most lists.
    k = 10
    xq = xb[:100]
    cpu_index.nprobe = min(cpu_index.nlist, 32)
    gpu_index.nprobe = cpu_index.nprobe

    print("Running baseline CPU and GPU searches before eviction...")
    D_cpu, I_cpu = cpu_index.search(xq, k)
    D_gpu0, I_gpu0 = gpu_index.search(xq, k)

    # Basic correctness check before eviction.
    np.testing.assert_array_equal(I_cpu, I_gpu0)
    np.testing.assert_array_almost_equal(D_cpu, D_gpu0, decimal=4)
    print("Baseline check passed: GPU results match CPU before eviction.")

    # Evict all currently loaded lists via the no-copy path. These lists
    # are not copied back from GPU to CPU; instead, we rely on the CPU index
    # as authoritative backing storage.
    print("Evicting loaded IVF lists using no-copy eviction...")
    reclaimed = faiss.evict_ivf_lists(gpu_index, load_ids)
    print("Reclaimed bytes per list (no-copy mode):")
    print(reclaimed)

    for list_id in load_ids:
        _require(
            not faiss.is_list_on_gpu(gpu_index, int(list_id)),
            f"List {list_id} should have been evicted from GPU",
        )

    evicted_after = set(faiss.get_evicted_lists(gpu_index).tolist())
    print(f"Evicted lists after no-copy eviction: {len(evicted_after)}")

    # Search after no-copy eviction: auto-fetch will reload any lists that
    # are needed from the CPU index on demand.
    print("Running GPU search after no-copy eviction (auto-fetch enabled)...")
    D_gpu1, I_gpu1 = gpu_index.search(xq, k)

    np.testing.assert_array_equal(I_cpu, I_gpu1)
    np.testing.assert_array_almost_equal(D_cpu, D_gpu1, decimal=4)
    print("Post-eviction check passed: GPU results still match CPU.")

    print("\nDemo complete.")
    print(
        "You can modify 'd', 'nb', 'nlist', or the fraction of loaded lists "
        "to experiment with different memory and performance trade-offs."
    )


if __name__ == "__main__":
    try:
        run_demo()
    except Exception as e:
        print(f"[demo_no_copy_evict_ivfflat] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
