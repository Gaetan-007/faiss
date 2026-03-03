#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark script for IVF-Flat GPU eviction with optional no-copy mode.

This benchmark is designed to be driven by `nsys profile`. When
`--use_profiler 1` is passed, it:

- wraps the main benchmark region (excluding warmup, IVF training and
  GPU index construction) with
  `torch.cuda.cudart().cudaProfilerStart()` /
  `torch.cuda.cudart().cudaProfilerStop()`, and
- uses NVTX ranges to distinguish the eviction and search phases within
  that region.

Two modes are supported:

- copy  (default):  standard copy-based eviction
- no_copy:         external CPU-backed no-copy eviction

Example usages (from the faiss repo root):

  # Copy-based eviction, with profiler markers
  nsys profile -o evict_copy_report \
      python3 demos/benchmark_ivfflat_evict.py --mode copy --use_profiler 1

  # No-copy eviction (requires GpuIndexIVFFlat with no-copy support)
  nsys profile -o evict_nocopy_report \
      python3 demos/benchmark_ivfflat_evict.py --mode no_copy --use_profiler 1
"""

import argparse
import sys

import numpy as np

import faiss


def _require(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def _check_faiss_gpu():
    if not hasattr(faiss, "get_num_gpus") or faiss.get_num_gpus() == 0:
        raise RuntimeError("GPU not available in this environment")


def _check_torch(use_profiler: bool):
    if not use_profiler:
        return None
    try:
        import torch  # type: ignore[import]
    except Exception as exc:
        raise RuntimeError(
            "use_profiler=1 requires PyTorch with CUDA support "
            "(import torch failed)"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is not available but use_profiler=1")
    return torch


def _build_cpu_ivfflat(d: int, nlist: int, nb: int, seed: int):
    rng = np.random.RandomState(seed)
    xb = rng.rand(nb, d).astype("float32")
    # Use enough training points to avoid k-means warnings like
    # "clustering M points to nlist centroids: please provide at least ...".
    # Empirically, ~40 * nlist points is sufficient for the default params.
    num_train = min(nb, max(200, nlist * 40))
    xt = xb[:num_train]

    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    index.train(xt)
    index.add(xb)
    return index, xb


def _choose_nonempty_lists(cpu_index, xq, nprobe: int):
    """Use CPU coarse quantizer to find IVF lists that will be visited."""
    _, list_ids = cpu_index.quantizer.search(xq, nprobe)
    ids = np.unique(list_ids.reshape(-1))
    non_empty = [
        int(i)
        for i in ids
        if 0 <= int(i) < cpu_index.nlist
        and cpu_index.invlists.list_size(int(i)) > 0
    ]
    return non_empty


def build_gpu_index_copy_mode(cpu_index, device: int):
    """Standard GPU clone: all lists reside on GPU, copy-based eviction."""
    res = faiss.StandardGpuResources()

    gpu_index = faiss.index_cpu_to_gpu(res, device, cpu_index)

    # Enable auto-fetch so evicted lists are fetched on-demand during search.
    if hasattr(faiss, "set_auto_fetch"):
        faiss.set_auto_fetch(gpu_index, True)

    return gpu_index, res


def build_gpu_index_no_copy_mode(cpu_index, device: int, load_fraction: float):
    """
    Build a GPU index that uses external CPU backing + no-copy eviction.

    Only a subset of IVF lists is loaded to GPU; remaining lists are
    kept on CPU as backing storage.
    """
    res = faiss.StandardGpuResources()

    d = cpu_index.d
    nlist = cpu_index.nlist

    gpu_index = faiss.GpuIndexIVFFlat(
        res, d, nlist, cpu_index.metric_type
    )

    # Choose some non-empty lists to initially load on GPU.
    invlists = cpu_index.invlists
    nonempty = [i for i in range(invlists.nlist) if invlists.list_size(i) > 0]
    _require(
        len(nonempty) > 0,
        "No non-empty IVF lists found; increase nb or adjust nlist",
    )

    n_load = max(1, int(len(nonempty) * load_fraction))
    load_ids = nonempty[:n_load]

    faiss.init_ivf_lists_from_cpu(gpu_index, cpu_index, load_ids)

    # Enable auto-fetch + no-copy eviction.
    faiss.set_auto_fetch(gpu_index, True)
    faiss.set_no_copy_evict(gpu_index, True)

    return gpu_index, res, load_ids


def run_benchmark(
    mode: str,
    use_profiler: bool,
    d: int,
    nlist: int,
    nb: int,
    nq: int,
    k: int,
    nprobe: int,
    warmup_search: int,
    warmup_evict: int,
):
    _check_faiss_gpu()
    torch = _check_torch(use_profiler)

    cpu_index, xb = _build_cpu_ivfflat(d=d, nlist=nlist, nb=nb, seed=123)
    print(
        f"CPU IndexIVFFlat: d={d}, nlist={nlist}, nb={nb}, "
        f"ntotal={cpu_index.ntotal}"
    )

    xq = xb[:nq]
    _require(xq.shape[0] == nq, "Not enough data points for queries")

    cpu_index.nprobe = nprobe

    device = 0

    if mode == "copy":
        gpu_index, res = build_gpu_index_copy_mode(cpu_index, device=device)
        evict_ids = _choose_nonempty_lists(cpu_index, xq, nprobe=nprobe)
        _require(
            len(evict_ids) > 0,
            "No candidate lists found for eviction in copy mode",
        )
    elif mode == "no_copy":
        gpu_index, res, evict_ids = build_gpu_index_no_copy_mode(
            cpu_index, device=device, load_fraction=0.5
        )
        _require(
            len(evict_ids) > 0,
            "No candidate lists found for eviction in no_copy mode",
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    gpu_index.nprobe = nprobe

    # Warmup search
    print(f"[warmup] search x{warmup_search}")
    for _ in range(warmup_search):
        gpu_index.search(xq, k)

    # Warmup evict/load to ensure JIT / caches etc. are primed
    evict_ids_arr = np.asarray(evict_ids, dtype="int64")
    print(f"[warmup] evict/load x{warmup_evict}, #lists={len(evict_ids_arr)}")
    for _ in range(warmup_evict):
        faiss.evict_ivf_lists(gpu_index, evict_ids_arr)
        faiss.load_ivf_lists(gpu_index, evict_ids_arr)

    # ---------------- Main profiled region ----------------
    # We exclude warmup, IVF training and GPU initialization from the
    # profiler window. Only the eviction and search benchmarks are
    # enclosed by cudaProfilerStart/Stop, with NVTX ranges marking
    # each phase.
    if torch is not None:
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()

    # ---------------- Eviction phase benchmark ----------------
    print("[benchmark] eviction phase")
    if torch is not None:
        torch.cuda.synchronize()
        # NVTX range for eviction
        torch.cuda.nvtx.range_push("evict_ivf_lists")

    reclaimed = faiss.evict_ivf_lists(gpu_index, evict_ids_arr)

    if torch is not None:
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

    print("Reclaimed bytes per list (first 8):", reclaimed[:8])

    # ---------------- Search phase benchmark ----------------
    print("[benchmark] search phase")
    if torch is not None:
        torch.cuda.synchronize()
        # NVTX range for search
        torch.cuda.nvtx.range_push("search_ivfflat")

    D_gpu, I_gpu = gpu_index.search(xq, k)

    if torch is not None:
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()

    print("Search completed. Example distances/indices:")
    print("D[0,:5] =", D_gpu[0, :5])
    print("I[0,:5] =", I_gpu[0, :5])


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="IVF-Flat GPU eviction benchmark (copy vs no-copy)."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="copy",
        choices=["copy", "no_copy"],
        help="Eviction mode: 'copy' (default) or 'no_copy'.",
    )
    parser.add_argument(
        "--use_profiler",
        type=int,
        default=0,
        help=(
            "If 1, wrap eviction/search with "
            "torch.cuda.cudart().cudaProfilerStart/Stop."
        ),
    )
    parser.add_argument("--d", type=int, default=1024, help="Vector dimension.")
    parser.add_argument("--nlist", type=int, default=256, help="IVF nlist.")
    parser.add_argument("--nb", type=int, default=50000, help="DB size.")
    parser.add_argument("--nq", type=int, default=512, help="Number of queries.")
    parser.add_argument("--k", type=int, default=10, help="Top-k.")
    parser.add_argument(
        "--nprobe", type=int, default=128, help="nprobe for IVF search."
    )
    parser.add_argument(
        "--warmup_search",
        type=int,
        default=3,
        help="Number of warmup search iterations.",
    )
    parser.add_argument(
        "--warmup_evict",
        type=int,
        default=1,
        help="Number of warmup evict+load iterations.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    run_benchmark(
        mode=args.mode,
        use_profiler=bool(args.use_profiler),
        d=args.d,
        nlist=args.nlist,
        nb=args.nb,
        nq=args.nq,
        k=args.k,
        nprobe=args.nprobe,
        warmup_search=args.warmup_search,
        warmup_evict=args.warmup_evict,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[benchmark_ivfflat_evict] ERROR: {e}", file=sys.stderr)
        sys.exit(1)

