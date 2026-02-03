#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Big-batch IVF-Flat GPU benchmark. IVF-Flat only; no contrib.big_batch_search.
Configurable: nb, dim, nlist; which IVF lists (and vectors) are on GPU via eviction.
"""

import argparse
import multiprocessing
import sys
import time
from typing import List, Optional, Set

import numpy as np

import faiss
from faiss import (
    evict_ivf_lists,
    get_auto_fetch_stats,
    get_evicted_lists,
    reset_auto_fetch_stats,
    set_auto_fetch,
)

# Global random seed for reproducibility (data, gpu-lists-num, etc.)
RANDOM_SEED = 42


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Big-batch IVF-Flat GPU benchmark. Configurable nb/dim/nlist and GPU list pre-config."
    )

    group = parser.add_argument_group("database and index")
    group.add_argument("--nb", type=int, default=100_000, help="Total vectors in database")
    group.add_argument("--dim", type=int, default=1024, help="Vector dimension")
    group.add_argument("--nlist", type=int, default=100, help="Number of IVF clusters")
    group.add_argument("--nq", type=int, default=-1, help="Number of queries (default: min(10000, nb))")
    group.add_argument("--k", type=int, default=10, help="k nearest neighbors per query")
    group.add_argument("--nprobe", type=int, default=10, help="Clusters to probe per query")
    group.add_argument(
        "--train-threads",
        type=int,
        default=-1,
        help="Number of threads for CPU training (default: all CPU cores, -1=auto)",
    )

    group = parser.add_argument_group("GPU list pre-configuration")
    group.add_argument(
        "--gpu-lists",
        type=str,
        default=None,
        help="Comma-separated list IDs to keep on GPU (e.g. '0,1,2'). Omit = all on GPU.",
    )
    group.add_argument(
        "--gpu-lists-num",
        type=int,
        default=-1,
        help=(
            "Number of random lists to keep on GPU (if >= 0 and --gpu-lists not set). "
            "Default -1 = all on GPU (or use --gpu-lists)."
        ),
    )
    group.add_argument(
        "--auto-fetch",
        action="store_true",
        help=(
            "Enable Faiss auto fetch: during search, automatically load evicted IVF lists "
            "from CPU cache to GPU when needed (page-fault style). Meaningful when some "
            "lists are evicted (--gpu-lists or --gpu-lists-num)."
        ),
    )

    group = parser.add_argument_group("GPU resources (StandardGpuResources)")
    group.add_argument("--gpu", type=int, default=0, help="GPU device id")
    group.add_argument("--gpu-device-mem-mb", type=int, default=-1, help="Device memory reservation MiB; -1=default")
    group.add_argument("--gpu-temp-mem-mb", type=int, default=-1, help="Temp memory MiB; -1=default")
    group.add_argument("--gpu-pinned-mem-mb", type=int, default=-1, help="Pinned memory MiB; -1=default")
    group.add_argument("--reserve-vecs", type=int, default=0, help="GpuClonerOptions.reserveVecs; 0=do not reserve")

    group = parser.add_argument_group("benchmark")
    group.add_argument("--nruns", type=int, default=5, help="Number of search runs for timing")
    group.add_argument("--no-cpu", action="store_true", help="Skip CPU reference search (default: run CPU comparison)")
    group.add_argument("--ref", action="store_true", help="Check correctness between GPU and CPU results (requires CPU search)")
    group.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Global random seed for reproducibility (default: %d)" % RANDOM_SEED,
    )

    return parser


def parse_gpu_lists(s: Optional[str], nlist: int) -> Optional[Set[int]]:
    """Parse --gpu-lists into a set of list IDs. None = all on GPU."""
    if s is None or s.strip() == "":
        return None
    raw = [int(x.strip()) for x in s.split(",") if x.strip()]
    ids = set(raw)
    for i in ids:
        if i < 0 or i >= nlist:
            raise ValueError(
                "gpu-lists contains list id %d but nlist=%d; valid ids 0..%d"
                % (i, nlist, nlist - 1)
            )
    return ids


def configure_gpu_resources(
    gpu: int,
    device_mem_mb: int,
    temp_mem_mb: int,
    pinned_mem_mb: int,
) -> faiss.StandardGpuResources:
    """Create and configure StandardGpuResources (before any device use)."""
    res = faiss.StandardGpuResources()

    def _bytes_from_mb(val_mb: int) -> int:
        if val_mb < 0:
            raise ValueError("memory value must be >= 0 or -1 for default, got %d" % val_mb)
        return val_mb * 1024 * 1024

    if device_mem_mb >= 0:
        if not hasattr(res, "setDeviceMemoryReservation"):
            raise RuntimeError(
                "StandardGpuResources.setDeviceMemoryReservation not available in this build"
            )
        res.setDeviceMemoryReservation(_bytes_from_mb(device_mem_mb))

    if temp_mem_mb >= 0:
        if not hasattr(res, "setTempMemory"):
            raise RuntimeError("StandardGpuResources.setTempMemory not available in this build")
        res.setTempMemory(_bytes_from_mb(temp_mem_mb))

    if pinned_mem_mb >= 0:
        if not hasattr(res, "setPinnedMemory"):
            raise RuntimeError(
                "StandardGpuResources.setPinnedMemory not available in this build"
            )
        res.setPinnedMemory(_bytes_from_mb(pinned_mem_mb))

    return res


def _progress_bar(current: int, total: int, bar_width: int = 40) -> None:
    """Print a single-line progress bar by percentage. Overwrites previous line."""
    if total <= 0:
        pct = 100.0
    else:
        pct = 100.0 * current / total
    filled = int(bar_width * current / total) if total > 0 else bar_width
    bar = "=" * filled + ">" * (1 if filled < bar_width else 0) + " " * (bar_width - filled - 1)
    sys.stdout.write("\rGenerating vectors: [%s] %5.1f%% (%d / %d)    " % (bar, pct, current, total))
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def _generate_vectors_with_progress(
    n_train: int,
    nb: int,
    nq: int,
    dim: int,
) -> tuple:
    """Generate x_train, xb, xq in chunks with percentage progress bar."""
    total_vecs = n_train + nb + nq
    chunk = max(1, min(500_000, total_vecs // 50))  # ~50 updates or 500k per chunk
    generated = 0

    x_train = np.empty((n_train, dim), dtype=np.float32)
    xb = np.empty((nb, dim), dtype=np.float32)
    xq = np.empty((nq, dim), dtype=np.float32)

    def fill_chunk(arr: np.ndarray, offset: int, count: int) -> None:
        if count <= 0:
            return
        arr[offset : offset + count] = np.random.randn(count, dim).astype(np.float32) * 0.1

    # Train
    pos = 0
    while pos < n_train:
        step = min(chunk, n_train - pos)
        fill_chunk(x_train, pos, step)
        pos += step
        generated += step
        _progress_bar(generated, total_vecs)

    # Database
    pos = 0
    while pos < nb:
        step = min(chunk, nb - pos)
        fill_chunk(xb, pos, step)
        pos += step
        generated += step
        _progress_bar(generated, total_vecs)

    # Queries
    pos = 0
    while pos < nq:
        step = min(chunk, nq - pos)
        fill_chunk(xq, pos, step)
        pos += step
        generated += step
        _progress_bar(generated, total_vecs)

    return x_train, xb, xq


def percentile(data: List[float], p: float) -> float:
    """Calculate percentile of a list of values."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    index = int(len(sorted_data) * p / 100.0)
    if index >= len(sorted_data):
        index = len(sorted_data) - 1
    return sorted_data[index]


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # Fix global random seed for reproducibility
    seed = args.seed
    np.random.seed(seed)
    print("Random seed: %d" % seed)

    nb = args.nb
    dim = args.dim
    nlist = args.nlist
    nprobe = args.nprobe
    k = args.k
    nq = args.nq if args.nq > 0 else min(10_000, nb)

    res = configure_gpu_resources(
        args.gpu,
        args.gpu_device_mem_mb,
        args.gpu_temp_mem_mb,
        args.gpu_pinned_mem_mb,
    )

    # Generate data in chunks with percentage progress bar (seed already set at start of main)
    n_train = max(nlist * 40, nb // 10)
    print("Generating random vectors (train=%d, db=%d, queries=%d, dim=%d)..." % (n_train, nb, nq, dim))
    x_train, xb, xq = _generate_vectors_with_progress(n_train, nb, nq, dim)

    # Set training threads
    if args.train_threads > 0:
        train_threads = args.train_threads
    else:
        # Default: use all CPU cores
        train_threads = multiprocessing.cpu_count()
    
    print("Setting training threads to %d" % train_threads)
    faiss.omp_set_num_threads(train_threads)

    print("Building CPU IVF-Flat index...")
    cpu_index = faiss.index_factory(dim, "IVF%d,Flat" % nlist, faiss.METRIC_L2)
    t0 = time.time()
    cpu_index.train(x_train)
    train_time = time.time() - t0
    t0 = time.time()
    cpu_index.add(xb)
    add_time = time.time() - t0
    print("  train: %.3f s (threads: %d)" % (train_time, train_threads))
    print("  add:   %.3f s" % add_time)
    print("  total: %.3f s" % (train_time + add_time))

    # Determine GPU lists: --gpu-lists takes precedence, then --gpu-lists-num, else all
    gpu_lists = parse_gpu_lists(args.gpu_lists, nlist)
    if gpu_lists is None and args.gpu_lists_num >= 0:
        if args.gpu_lists_num > nlist:
            raise ValueError(
                "gpu-lists-num (%d) cannot exceed nlist (%d)" % (args.gpu_lists_num, nlist)
            )
        # Use same global seed for reproducible random list selection
        rng = np.random.RandomState(seed)
        gpu_lists = set(rng.choice(nlist, size=args.gpu_lists_num, replace=False).tolist())
        print("gpu-lists (randomly selected %d lists): %s" % (args.gpu_lists_num, sorted(gpu_lists)))
    elif gpu_lists is not None:
        print("gpu-lists (keep on GPU):", sorted(gpu_lists))
    else:
        print("gpu-lists: all (default)")

    co = faiss.GpuClonerOptions()
    if args.reserve_vecs > 0:
        co.reserveVecs = args.reserve_vecs
    print("Cloning to GPU %d (reserveVecs=%s)..." % (args.gpu, co.reserveVecs or "0"))
    t0 = time.time()
    gpu_index = faiss.index_cpu_to_gpu(res, args.gpu, cpu_index, co)
    gpu_index.nprobe = nprobe
    print("  clone: %.3f s" % (time.time() - t0))

    if gpu_lists is not None:
        to_evict = [i for i in range(nlist) if i not in gpu_lists]
        if to_evict:
            print(
                "Evicting %d lists to CPU (keeping %d on GPU)..."
                % (len(to_evict), len(gpu_lists))
            )
            t0 = time.time()
            evict_ivf_lists(gpu_index, to_evict)
            print("  evict: %.3f s" % (time.time() - t0))
        evicted = get_evicted_lists(gpu_index)
        print("  evicted list count: %d" % len(evicted))

    # Enable auto fetch (swap-in/swap-out) if requested
    if args.auto_fetch:
        if not hasattr(faiss, "set_auto_fetch"):
            raise RuntimeError(
                "Auto-fetch not available in this build (set_auto_fetch not found)"
            )
        set_auto_fetch(gpu_index, True)
        print("Auto-fetch enabled (evicted lists will be loaded on demand during search).")

    print(
        "Running GPU search (nq=%d, k=%d, nprobe=%d, nruns=%d)..."
        % (nq, k, nprobe, args.nruns)
    )
    if args.auto_fetch:
        try:
            reset_auto_fetch_stats(gpu_index)
        except AttributeError:
            pass  # older build may not have reset_auto_fetch_stats
    gpu_times = []
    for _ in range(args.nruns):
        t0 = time.time()
        D_gpu, I_gpu = gpu_index.search(xq, k)
        elapsed = time.time() - t0
        gpu_times.append(elapsed)
    
    gpu_avg = sum(gpu_times) / len(gpu_times)
    gpu_min = min(gpu_times)
    gpu_max = max(gpu_times)
    gpu_p50 = percentile(gpu_times, 50)
    gpu_p95 = percentile(gpu_times, 95)
    gpu_p99 = percentile(gpu_times, 99)
    
    print("  GPU search timing:")
    print("    avg:  %.3f s (qps: %.0f)" % (gpu_avg, nq / gpu_avg))
    print("    min:  %.3f s (qps: %.0f)" % (gpu_min, nq / gpu_min))
    print("    max:  %.3f s" % gpu_max)
    print("    P50:  %.3f s" % gpu_p50)
    print("    P95:  %.3f s" % gpu_p95)
    print("    P99:  %.3f s" % gpu_p99)

    if args.auto_fetch:
        try:
            stats = get_auto_fetch_stats(gpu_index)
            print("  Auto-fetch stats (over %d runs):" % args.nruns)
            print(
                "    total_fetches: %d, total_lists_fetched: %d, total_bytes_fetched: %d"
                % (
                    stats["total_fetches"],
                    stats["total_lists_fetched"],
                    stats["total_bytes_fetched"],
                )
            )
        except Exception as e:
            print("  Auto-fetch stats unavailable: %s" % e)

    # CPU reference search (default unless --no-cpu)
    cpu_t_ref = None
    if not args.no_cpu:
        cpu_index.nprobe = nprobe
        print("Running CPU reference search...")
        t0 = time.time()
        D_ref, I_ref = cpu_index.search(xq, k)
        cpu_t_ref = time.time() - t0
        print("  CPU search: %.3f s (qps: %.0f)" % (cpu_t_ref, nq / cpu_t_ref))
        
        # Correctness check if requested
        if args.ref:
            if I_gpu.size != I_ref.size:
                raise AssertionError(
                    "GPU and CPU result size mismatch: %d vs %d" % (I_gpu.size, I_ref.size)
                )
            mismatch = np.sum(I_gpu != I_ref)
            if mismatch > 0:
                raise AssertionError("GPU vs CPU label mismatch: %d / %d" % (mismatch, I_ref.size))
            try:
                np.testing.assert_almost_equal(D_gpu, D_ref, decimal=4)
            except AssertionError as e:
                raise AssertionError("GPU vs CPU distance mismatch") from e
            print("  correctness: OK")
        
        # Speedup comparison
        if cpu_t_ref is not None:
            speedup_avg = cpu_t_ref / gpu_avg
            speedup_min = cpu_t_ref / gpu_min
            print("  Speedup vs CPU:")
            print("    vs avg: %.2fx" % speedup_avg)
            print("    vs min: %.2fx" % speedup_min)


if __name__ == "__main__":
    main()

