#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
GPU Memory Pool Resize Overhead Benchmark

This benchmark measures the overhead of dynamic memory pool resizing operations
(expand/shrink) in FAISS GPU, including:

1. Pure resize operations (stepping through different resize amounts)
2. Concurrent search + resize operations (stepping through batch sizes and resize amounts)

Results are visualized as charts showing latency breakdown.
"""

import argparse
import gc
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for saving figures
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import faiss

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
    except ImportError as exc:
        raise ImportError(
            "GpuPoolController not found. Make sure faiss-gpu is built with IPC support."
        ) from exc


# ============================================================================
# Data Classes for Results
# ============================================================================

@dataclass
class ResizeResult:
    """Result of a single resize operation."""
    operation: str  # "expand" or "shrink"
    target_bytes: int
    delta_bytes: int  # Amount of memory changed
    latency_ms: float
    status: int
    actual_size_before: int
    actual_size_after: int
    available_before: int
    available_after: int


@dataclass
class SearchResult:
    """Result of a single search operation."""
    nq: int
    k: int
    latency_ms: float
    qps: float


@dataclass
class ConcurrentResult:
    """Result of concurrent search + resize."""
    search_nq: int
    resize_delta_bytes: int
    resize_operation: str
    search_latency_ms: float
    resize_latency_ms: float
    total_latency_ms: float
    resize_status: int
    # Additional timing for detailed analysis
    search_start_ms: float = 0.0  # Relative to total_start
    search_end_ms: float = 0.0
    resize_start_ms: float = 0.0
    resize_end_ms: float = 0.0
    # Pure resize overhead (from Benchmark 1, for comparison)
    pure_resize_latency_ms: float = 0.0


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark."""
    device_id: int = 0
    dim: int = 128
    nb: int = 100_000
    nlist: int = 100
    nprobe: int = 10
    k: int = 10
    base_pool_size_mb: int = 256
    nruns: int = 5
    warmup_runs: int = 2
    output_dir: str = "."


# ============================================================================
# Helper Functions
# ============================================================================

def bytes_to_mb(b: int) -> float:
    return b / (1024 * 1024)


def mb_to_bytes(mb: float) -> int:
    return int(mb * 1024 * 1024)


def format_bytes(b: int) -> str:
    """Format bytes as human-readable string."""
    if b >= 1024 * 1024 * 1024:
        return f"{b / (1024**3):.2f} GB"
    elif b >= 1024 * 1024:
        return f"{b / (1024**2):.2f} MB"
    elif b >= 1024:
        return f"{b / 1024:.2f} KB"
    else:
        return f"{b} B"


def percentile(data: List[float], p: float) -> float:
    """Calculate percentile of a list of values."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    index = int(len(sorted_data) * p / 100.0)
    if index >= len(sorted_data):
        index = len(sorted_data) - 1
    return sorted_data[index]


def stats_summary(data: List[float]) -> Dict[str, float]:
    """Calculate summary statistics."""
    if not data:
        return {"avg": 0, "min": 0, "max": 0, "p50": 0, "p95": 0, "p99": 0}
    return {
        "avg": sum(data) / len(data),
        "min": min(data),
        "max": max(data),
        "p50": percentile(data, 50),
        "p95": percentile(data, 95),
        "p99": percentile(data, 99),
    }


# ============================================================================
# Benchmark Setup
# ============================================================================

def create_gpu_index_and_resources(
    cfg: BenchmarkConfig,
) -> Tuple[faiss.StandardGpuResources, "faiss.GpuIndexIVFFlat", np.ndarray]:
    """Create GPU index with PreallocMemoryPool enabled."""
    print(f"Creating GPU resources with {cfg.base_pool_size_mb} MB reservation...")
    
    res = faiss.StandardGpuResources()
    
    # IMPORTANT: When using setDeviceMemoryReservation, ALL memory (including temp)
    # must fit within the preallocated pool. Default temp memory can be up to 1.5 GB,
    # which may exceed the pool size. Set temp memory to a fraction of the pool.
    # Use ~1/4 of pool for temp memory, leaving the rest for index data.
    temp_mem_mb = max(64, cfg.base_pool_size_mb // 4)
    print(f"Setting temp memory to {temp_mem_mb} MB (1/4 of pool)")
    res.setTempMemory(mb_to_bytes(temp_mem_mb))
    res.setDeviceMemoryReservation(mb_to_bytes(cfg.base_pool_size_mb))
    
    print(f"Generating random vectors (nb={cfg.nb}, dim={cfg.dim})...")
    np.random.seed(42)
    xb = np.random.randn(cfg.nb, cfg.dim).astype(np.float32) * 0.1
    
    print(f"Building CPU IVF-Flat index (nlist={cfg.nlist})...")
    cpu_index = faiss.index_factory(cfg.dim, f"IVF{cfg.nlist},Flat", faiss.METRIC_L2)
    n_train = max(cfg.nlist * 40, cfg.nb // 10)
    x_train = np.random.randn(n_train, cfg.dim).astype(np.float32) * 0.1
    cpu_index.train(x_train)
    cpu_index.add(xb)
    
    print(f"Cloning to GPU {cfg.device_id}...")
    co = faiss.GpuClonerOptions()
    gpu_index = faiss.index_cpu_to_gpu(res, cfg.device_id, cpu_index, co)
    gpu_index.nprobe = cfg.nprobe
    
    return res, gpu_index, xb


def warmup_operations(
    gpu_index: "faiss.GpuIndexIVFFlat",
    ctrl: GpuPoolController,
    xq: np.ndarray,
    k: int,
    nruns: int = 2,
) -> None:
    """Warm up GPU and IPC operations."""
    print("Warming up GPU operations...")
    for _ in range(nruns):
        gpu_index.search(xq[:100], k)
        ctrl.query()


# ============================================================================
# Benchmark 1: Pure Resize Operations
# ============================================================================

def bench_pure_resize(
    cfg: BenchmarkConfig,
    ctrl: GpuPoolController,
    delta_steps_mb: List[int],
) -> Dict[str, List[ResizeResult]]:
    """
    Benchmark pure resize operations (expand/shrink) with different delta sizes.
    
    Args:
        cfg: Benchmark configuration
        ctrl: GPU pool controller
        delta_steps_mb: List of delta sizes in MB to test
        
    Returns:
        Dictionary with "expand" and "shrink" keys, each containing list of ResizeResult
    """
    results = {"expand": [], "shrink": []}
    
    # Get initial state
    initial = ctrl.query()
    initial_size = initial["actual_size"]
    print(f"Initial pool size: {format_bytes(initial_size)}")
    
    for delta_mb in delta_steps_mb:
        delta_bytes = mb_to_bytes(delta_mb)
        
        # Reset to initial size before each test
        ctrl.shrink(initial_size)
        time.sleep(0.1)  # Allow pool to settle
        
        # --- Expand Test ---
        expand_latencies = []
        for run in range(cfg.nruns):
            # Reset before each expand
            current = ctrl.query()
            before_size = current["actual_size"]
            before_available = current["available"]
            target_size = before_size + delta_bytes
            
            t0 = time.perf_counter()
            result = ctrl.expand(target_size, timeout_ms=30000)
            t1 = time.perf_counter()
            
            latency_ms = (t1 - t0) * 1000
            expand_latencies.append(latency_ms)
            
            if run == 0:  # Record first run details
                expand_result = ResizeResult(
                    operation="expand",
                    target_bytes=target_size,
                    delta_bytes=delta_bytes,
                    latency_ms=latency_ms,
                    status=int(result["status"]),
                    actual_size_before=before_size,
                    actual_size_after=result["actual_size"],
                    available_before=before_available,
                    available_after=result["available"],
                )
            
            # Shrink back for next expand run
            if run < cfg.nruns - 1:
                ctrl.shrink(initial_size)
                time.sleep(0.05)
        
        # Use average latency
        expand_result.latency_ms = sum(expand_latencies) / len(expand_latencies)
        results["expand"].append(expand_result)
        
        print(f"  Expand +{delta_mb} MB: "
              f"avg={expand_result.latency_ms:.2f} ms, "
              f"status={ResizeStatus(expand_result.status).name}")
        
        # --- Shrink Test ---
        # First expand to have something to shrink
        expanded_size = initial_size + delta_bytes
        ctrl.expand(expanded_size)
        time.sleep(0.1)
        
        shrink_latencies = []
        for run in range(cfg.nruns):
            # Re-expand if needed
            ctrl.expand(expanded_size)
            time.sleep(0.05)
            
            current = ctrl.query()
            before_size = current["actual_size"]
            before_available = current["available"]
            
            t0 = time.perf_counter()
            result = ctrl.shrink(initial_size, timeout_ms=30000)
            t1 = time.perf_counter()
            
            latency_ms = (t1 - t0) * 1000
            shrink_latencies.append(latency_ms)
            
            if run == 0:
                shrink_result = ResizeResult(
                    operation="shrink",
                    target_bytes=initial_size,
                    delta_bytes=delta_bytes,
                    latency_ms=latency_ms,
                    status=int(result["status"]),
                    actual_size_before=before_size,
                    actual_size_after=result["actual_size"],
                    available_before=before_available,
                    available_after=result["available"],
                )
        
        shrink_result.latency_ms = sum(shrink_latencies) / len(shrink_latencies)
        results["shrink"].append(shrink_result)
        
        print(f"  Shrink -{delta_mb} MB: "
              f"avg={shrink_result.latency_ms:.2f} ms, "
              f"status={ResizeStatus(shrink_result.status).name}")
    
    # Reset to initial size
    ctrl.shrink(initial_size)
    
    return results


# ============================================================================
# Benchmark 2: Concurrent Search + Resize
# ============================================================================

def bench_concurrent_search_resize(
    cfg: BenchmarkConfig,
    gpu_index: "faiss.GpuIndexIVFFlat",
    ctrl: GpuPoolController,
    xq: np.ndarray,
    batch_sizes: List[int],
    delta_steps_mb: List[int],
    pure_resize_results: Optional[Dict[str, List[ResizeResult]]] = None,
) -> List[ConcurrentResult]:
    """
    Benchmark concurrent search and resize operations.
    
    This simulates a realistic scenario where search requests and resize commands
    arrive simultaneously.
    
    IMPORTANT: The resize operation internally calls cudaDeviceSynchronize() which
    blocks until all GPU kernels (including search) complete. Therefore:
    - resize_latency includes: IPC overhead + waiting for search + actual resize
    - To get pure resize overhead, compare with Benchmark 1 results
    
    Args:
        cfg: Benchmark configuration
        gpu_index: GPU IVF Flat index
        ctrl: GPU pool controller
        xq: Query vectors
        batch_sizes: List of query batch sizes to test
        delta_steps_mb: List of resize delta sizes in MB
        pure_resize_results: Results from Benchmark 1 for comparison
        
    Returns:
        List of ConcurrentResult
    """
    results = []
    
    # Build lookup for pure resize latencies
    pure_resize_lookup = {}
    if pure_resize_results:
        for op in ["expand", "shrink"]:
            for r in pure_resize_results.get(op, []):
                delta_mb = r.delta_bytes // (1024 * 1024)
                pure_resize_lookup[(op, delta_mb)] = r.latency_ms
    
    # Get initial state
    initial = ctrl.query()
    initial_size = initial["actual_size"]
    
    for nq in batch_sizes:
        for delta_mb in delta_steps_mb:
            delta_bytes = mb_to_bytes(delta_mb)
            
            # Test both expand and shrink during search
            for op in ["expand", "shrink"]:
                # Prepare: set pool to appropriate state
                if op == "expand":
                    ctrl.shrink(initial_size)
                    target_size = initial_size + delta_bytes
                else:
                    ctrl.expand(initial_size + delta_bytes)
                    target_size = initial_size
                time.sleep(0.1)
                
                concurrent_results = []
                
                for run in range(cfg.nruns):
                    # Reset state before each run
                    if op == "expand":
                        ctrl.shrink(initial_size)
                    else:
                        ctrl.expand(initial_size + delta_bytes)
                    time.sleep(0.05)
                    
                    # Shared results container with detailed timing
                    # [latency_ms, start_time, end_time]
                    search_timing = [0.0, 0.0, 0.0]
                    resize_timing = [0.0, 0.0, 0.0]
                    resize_status = [0]
                    base_time = [0.0]  # Reference time for relative timestamps
                    
                    def search_thread():
                        t0 = time.perf_counter()
                        search_timing[1] = (t0 - base_time[0]) * 1000  # start relative
                        gpu_index.search(xq[:nq], cfg.k)
                        t1 = time.perf_counter()
                        search_timing[2] = (t1 - base_time[0]) * 1000  # end relative
                        search_timing[0] = (t1 - t0) * 1000  # latency
                    
                    def resize_thread():
                        t0 = time.perf_counter()
                        resize_timing[1] = (t0 - base_time[0]) * 1000  # start relative
                        if op == "expand":
                            result = ctrl.expand(target_size, timeout_ms=30000)
                        else:
                            result = ctrl.shrink(target_size, timeout_ms=30000)
                        t1 = time.perf_counter()
                        resize_timing[2] = (t1 - base_time[0]) * 1000  # end relative
                        resize_timing[0] = (t1 - t0) * 1000  # latency
                        resize_status[0] = int(result["status"])
                    
                    # Launch threads concurrently
                    t_search = threading.Thread(target=search_thread)
                    t_resize = threading.Thread(target=resize_thread)
                    
                    base_time[0] = time.perf_counter()
                    t_search.start()
                    t_resize.start()
                    t_search.join()
                    t_resize.join()
                    t_total_end = time.perf_counter()
                    
                    total_latency = (t_total_end - base_time[0]) * 1000
                    
                    concurrent_results.append(ConcurrentResult(
                        search_nq=nq,
                        resize_delta_bytes=delta_bytes,
                        resize_operation=op,
                        search_latency_ms=search_timing[0],
                        resize_latency_ms=resize_timing[0],
                        total_latency_ms=total_latency,
                        resize_status=resize_status[0],
                        search_start_ms=search_timing[1],
                        search_end_ms=search_timing[2],
                        resize_start_ms=resize_timing[1],
                        resize_end_ms=resize_timing[2],
                        pure_resize_latency_ms=pure_resize_lookup.get((op, delta_mb), 0.0),
                    ))
                
                # Average results
                avg_result = ConcurrentResult(
                    search_nq=nq,
                    resize_delta_bytes=delta_bytes,
                    resize_operation=op,
                    search_latency_ms=sum(r.search_latency_ms for r in concurrent_results) / len(concurrent_results),
                    resize_latency_ms=sum(r.resize_latency_ms for r in concurrent_results) / len(concurrent_results),
                    total_latency_ms=sum(r.total_latency_ms for r in concurrent_results) / len(concurrent_results),
                    resize_status=concurrent_results[0].resize_status,
                    search_start_ms=sum(r.search_start_ms for r in concurrent_results) / len(concurrent_results),
                    search_end_ms=sum(r.search_end_ms for r in concurrent_results) / len(concurrent_results),
                    resize_start_ms=sum(r.resize_start_ms for r in concurrent_results) / len(concurrent_results),
                    resize_end_ms=sum(r.resize_end_ms for r in concurrent_results) / len(concurrent_results),
                    pure_resize_latency_ms=pure_resize_lookup.get((op, delta_mb), 0.0),
                )
                results.append(avg_result)
                
                # Calculate blocking time: resize waits for search due to cudaDeviceSynchronize
                # If resize_end > search_end and resize started before search ended,
                # then blocking_time ≈ search_end - resize_start (time resize waited for search)
                blocking_time = max(0, min(avg_result.search_end_ms, avg_result.resize_end_ms) 
                                      - max(avg_result.search_start_ms, avg_result.resize_start_ms))
                pure_resize = avg_result.pure_resize_latency_ms
                
                print(f"  nq={nq}, {op} {delta_mb} MB: "
                      f"search={avg_result.search_latency_ms:.2f} ms, "
                      f"resize={avg_result.resize_latency_ms:.2f} ms "
                      f"(pure={pure_resize:.2f}, blocked={avg_result.resize_latency_ms - pure_resize:.2f}), "
                      f"total={avg_result.total_latency_ms:.2f} ms")
    
    # Reset to initial size
    ctrl.shrink(initial_size)
    
    return results


# ============================================================================
# Baseline: Search-only latency
# ============================================================================

def bench_search_baseline(
    cfg: BenchmarkConfig,
    gpu_index: "faiss.GpuIndexIVFFlat",
    xq: np.ndarray,
    batch_sizes: List[int],
) -> Dict[int, float]:
    """
    Benchmark search-only latency as baseline.
    
    Returns:
        Dictionary mapping batch size to average latency in ms
    """
    results = {}
    
    for nq in batch_sizes:
        latencies = []
        for _ in range(cfg.nruns):
            t0 = time.perf_counter()
            gpu_index.search(xq[:nq], cfg.k)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)
        
        avg_latency = sum(latencies) / len(latencies)
        results[nq] = avg_latency
        print(f"  Search nq={nq}: avg={avg_latency:.2f} ms")
    
    return results


# ============================================================================
# Visualization
# ============================================================================

def plot_pure_resize_results(
    results: Dict[str, List[ResizeResult]],
    output_path: str,
) -> None:
    """Plot pure resize benchmark results."""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping plot generation")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Extract data
    expand_deltas = [r.delta_bytes / (1024**2) for r in results["expand"]]
    expand_latencies = [r.latency_ms for r in results["expand"]]
    shrink_deltas = [r.delta_bytes / (1024**2) for r in results["shrink"]]
    shrink_latencies = [r.latency_ms for r in results["shrink"]]
    
    # Plot 1: Latency vs Delta Size
    ax1 = axes[0]
    ax1.bar(np.arange(len(expand_deltas)) - 0.2, expand_latencies, 
            width=0.4, label='Expand', color='#2ecc71', alpha=0.8)
    ax1.bar(np.arange(len(shrink_deltas)) + 0.2, shrink_latencies, 
            width=0.4, label='Shrink', color='#e74c3c', alpha=0.8)
    ax1.set_xlabel('Delta Size (MB)')
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('Resize Latency vs Delta Size')
    ax1.set_xticks(np.arange(len(expand_deltas)))
    ax1.set_xticklabels([f'{d:.0f}' for d in expand_deltas])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Latency per MB
    ax2 = axes[1]
    expand_per_mb = [l / d if d > 0 else 0 for l, d in zip(expand_latencies, expand_deltas)]
    shrink_per_mb = [l / d if d > 0 else 0 for l, d in zip(shrink_latencies, shrink_deltas)]
    
    ax2.plot(expand_deltas, expand_per_mb, 'o-', label='Expand', color='#2ecc71', linewidth=2, markersize=8)
    ax2.plot(shrink_deltas, shrink_per_mb, 's-', label='Shrink', color='#e74c3c', linewidth=2, markersize=8)
    ax2.set_xlabel('Delta Size (MB)')
    ax2.set_ylabel('Latency per MB (ms/MB)')
    ax2.set_title('Resize Efficiency (Latency per MB)')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved pure resize plot to: {output_path}")


def plot_concurrent_results(
    concurrent_results: List[ConcurrentResult],
    search_baseline: Dict[int, float],
    output_path: str,
) -> None:
    """Plot concurrent search + resize benchmark results."""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping plot generation")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Separate expand and shrink results
    expand_results = [r for r in concurrent_results if r.resize_operation == "expand"]
    shrink_results = [r for r in concurrent_results if r.resize_operation == "shrink"]
    
    # Get unique batch sizes and delta sizes
    batch_sizes = sorted(set(r.search_nq for r in concurrent_results))
    delta_sizes = sorted(set(r.resize_delta_bytes for r in concurrent_results))
    delta_sizes_mb = [d / (1024**2) for d in delta_sizes]
    
    # Plot 1: Search latency with expand (heatmap-style)
    ax1 = axes[0, 0]
    search_lat_matrix = np.zeros((len(batch_sizes), len(delta_sizes)))
    for r in expand_results:
        i = batch_sizes.index(r.search_nq)
        j = delta_sizes.index(r.resize_delta_bytes)
        search_lat_matrix[i, j] = r.search_latency_ms
    
    im1 = ax1.imshow(search_lat_matrix, aspect='auto', cmap='YlOrRd')
    ax1.set_xlabel('Expand Delta (MB)')
    ax1.set_ylabel('Batch Size (nq)')
    ax1.set_title('Search Latency During Expand (ms)')
    ax1.set_xticks(np.arange(len(delta_sizes_mb)))
    ax1.set_xticklabels([f'{d:.0f}' for d in delta_sizes_mb])
    ax1.set_yticks(np.arange(len(batch_sizes)))
    ax1.set_yticklabels([str(b) for b in batch_sizes])
    plt.colorbar(im1, ax=ax1)
    
    # Annotate cells
    for i in range(len(batch_sizes)):
        for j in range(len(delta_sizes)):
            ax1.text(j, i, f'{search_lat_matrix[i, j]:.1f}', 
                    ha='center', va='center', fontsize=8,
                    color='white' if search_lat_matrix[i, j] > search_lat_matrix.max() * 0.6 else 'black')
    
    # Plot 2: Search latency with shrink
    ax2 = axes[0, 1]
    search_lat_matrix_shrink = np.zeros((len(batch_sizes), len(delta_sizes)))
    for r in shrink_results:
        i = batch_sizes.index(r.search_nq)
        j = delta_sizes.index(r.resize_delta_bytes)
        search_lat_matrix_shrink[i, j] = r.search_latency_ms
    
    im2 = ax2.imshow(search_lat_matrix_shrink, aspect='auto', cmap='YlOrRd')
    ax2.set_xlabel('Shrink Delta (MB)')
    ax2.set_ylabel('Batch Size (nq)')
    ax2.set_title('Search Latency During Shrink (ms)')
    ax2.set_xticks(np.arange(len(delta_sizes_mb)))
    ax2.set_xticklabels([f'{d:.0f}' for d in delta_sizes_mb])
    ax2.set_yticks(np.arange(len(batch_sizes)))
    ax2.set_yticklabels([str(b) for b in batch_sizes])
    plt.colorbar(im2, ax=ax2)
    
    for i in range(len(batch_sizes)):
        for j in range(len(delta_sizes)):
            ax2.text(j, i, f'{search_lat_matrix_shrink[i, j]:.1f}', 
                    ha='center', va='center', fontsize=8,
                    color='white' if search_lat_matrix_shrink[i, j] > search_lat_matrix_shrink.max() * 0.6 else 'black')
    
    # Plot 3: Search latency overhead vs baseline (grouped bar)
    ax3 = axes[1, 0]
    x = np.arange(len(batch_sizes))
    width = 0.25
    
    baseline_lats = [search_baseline.get(b, 0) for b in batch_sizes]
    
    # Average across delta sizes for each batch size
    expand_avg_lats = []
    shrink_avg_lats = []
    for b in batch_sizes:
        expand_lats = [r.search_latency_ms for r in expand_results if r.search_nq == b]
        shrink_lats = [r.search_latency_ms for r in shrink_results if r.search_nq == b]
        expand_avg_lats.append(sum(expand_lats) / len(expand_lats) if expand_lats else 0)
        shrink_avg_lats.append(sum(shrink_lats) / len(shrink_lats) if shrink_lats else 0)
    
    ax3.bar(x - width, baseline_lats, width, label='Baseline', color='#3498db', alpha=0.8)
    ax3.bar(x, expand_avg_lats, width, label='+ Expand', color='#2ecc71', alpha=0.8)
    ax3.bar(x + width, shrink_avg_lats, width, label='+ Shrink', color='#e74c3c', alpha=0.8)
    ax3.set_xlabel('Batch Size (nq)')
    ax3.set_ylabel('Search Latency (ms)')
    ax3.set_title('Search Latency: Baseline vs Concurrent Resize')
    ax3.set_xticks(x)
    ax3.set_xticklabels([str(b) for b in batch_sizes])
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Overhead percentage
    ax4 = axes[1, 1]
    expand_overhead = [(e - b) / b * 100 if b > 0 else 0 for e, b in zip(expand_avg_lats, baseline_lats)]
    shrink_overhead = [(s - b) / b * 100 if b > 0 else 0 for s, b in zip(shrink_avg_lats, baseline_lats)]
    
    ax4.plot(batch_sizes, expand_overhead, 'o-', label='Expand Overhead', color='#2ecc71', linewidth=2, markersize=8)
    ax4.plot(batch_sizes, shrink_overhead, 's-', label='Shrink Overhead', color='#e74c3c', linewidth=2, markersize=8)
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Batch Size (nq)')
    ax4.set_ylabel('Overhead (%)')
    ax4.set_title('Search Latency Overhead from Concurrent Resize')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved concurrent results plot to: {output_path}")


def plot_resize_latency_breakdown(
    concurrent_results: List[ConcurrentResult],
    output_path: str,
) -> None:
    """Plot resize latency breakdown during concurrent operations.
    
    Shows the breakdown of resize latency into:
    - Pure resize overhead (from Benchmark 1)
    - Blocked time (waiting for search to complete due to cudaDeviceSynchronize)
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping plot generation")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Get unique values
    batch_sizes = sorted(set(r.search_nq for r in concurrent_results))
    delta_sizes = sorted(set(r.resize_delta_bytes for r in concurrent_results))
    delta_sizes_mb = [d / (1024**2) for d in delta_sizes]
    
    expand_results = [r for r in concurrent_results if r.resize_operation == "expand"]
    shrink_results = [r for r in concurrent_results if r.resize_operation == "shrink"]
    
    # Plot 1: Resize latency breakdown by batch size (stacked bar: pure + blocked)
    ax1 = axes[0, 0]
    
    # For shrink operations (which have cudaDeviceSynchronize blocking)
    x = np.arange(len(batch_sizes))
    width = 0.35
    
    # Get pure resize latency (first result has it, same for all with same delta)
    first_delta = delta_sizes[len(delta_sizes) // 2]  # Use middle delta size
    first_delta_mb = first_delta / (1024**2)
    
    shrink_pure = []
    shrink_blocked = []
    expand_pure = []
    expand_blocked = []
    
    for b in batch_sizes:
        # Get shrink results for this batch size (average across delta sizes)
        shrink_r = [r for r in shrink_results if r.search_nq == b]
        expand_r = [r for r in expand_results if r.search_nq == b]
        
        if shrink_r:
            avg_resize = sum(r.resize_latency_ms for r in shrink_r) / len(shrink_r)
            avg_pure = sum(r.pure_resize_latency_ms for r in shrink_r) / len(shrink_r)
            shrink_pure.append(avg_pure)
            shrink_blocked.append(max(0, avg_resize - avg_pure))
        else:
            shrink_pure.append(0)
            shrink_blocked.append(0)
            
        if expand_r:
            avg_resize = sum(r.resize_latency_ms for r in expand_r) / len(expand_r)
            avg_pure = sum(r.pure_resize_latency_ms for r in expand_r) / len(expand_r)
            expand_pure.append(avg_pure)
            expand_blocked.append(max(0, avg_resize - avg_pure))
        else:
            expand_pure.append(0)
            expand_blocked.append(0)
    
    # Stacked bar for shrink
    ax1.bar(x - width/2, shrink_pure, width, label='Shrink: Pure Resize', color='#c0392b', alpha=0.9)
    ax1.bar(x - width/2, shrink_blocked, width, bottom=shrink_pure, 
            label='Shrink: Blocked (cudaSync)', color='#e74c3c', alpha=0.5, hatch='//')
    # Stacked bar for expand
    ax1.bar(x + width/2, expand_pure, width, label='Expand: Pure Resize', color='#27ae60', alpha=0.9)
    ax1.bar(x + width/2, expand_blocked, width, bottom=expand_pure, 
            label='Expand: Blocked (cudaSync)', color='#2ecc71', alpha=0.5, hatch='\\\\')
    
    ax1.set_xlabel('Batch Size (nq)')
    ax1.set_ylabel('Resize Latency (ms)')
    ax1.set_title('Resize Latency Breakdown: Pure vs Blocked Time')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(b) for b in batch_sizes])
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Blocked percentage by batch size
    ax2 = axes[0, 1]
    shrink_blocked_pct = [b / (p + b) * 100 if (p + b) > 0 else 0 
                          for p, b in zip(shrink_pure, shrink_blocked)]
    expand_blocked_pct = [b / (p + b) * 100 if (p + b) > 0 else 0 
                          for p, b in zip(expand_pure, expand_blocked)]
    
    ax2.plot(batch_sizes, shrink_blocked_pct, 's-', label='Shrink', color='#e74c3c', linewidth=2, markersize=8)
    ax2.plot(batch_sizes, expand_blocked_pct, 'o-', label='Expand', color='#2ecc71', linewidth=2, markersize=8)
    ax2.set_xlabel('Batch Size (nq)')
    ax2.set_ylabel('Blocked Time Percentage (%)')
    ax2.set_title('Percentage of Resize Time Spent Waiting for Search')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 100)
    
    # Plot 3: Timeline visualization for one batch size
    ax3 = axes[1, 0]
    mid_batch = batch_sizes[len(batch_sizes) // 2]
    mid_delta = delta_sizes[len(delta_sizes) // 2]
    
    # Get one representative result
    rep_shrink = [r for r in shrink_results if r.search_nq == mid_batch and r.resize_delta_bytes == mid_delta]
    rep_expand = [r for r in expand_results if r.search_nq == mid_batch and r.resize_delta_bytes == mid_delta]
    
    y_pos = 0
    bar_height = 0.3
    colors = {'search': '#3498db', 'resize': '#e74c3c', 'expand': '#2ecc71'}
    
    if rep_shrink:
        r = rep_shrink[0]
        # Search bar
        ax3.barh(y_pos, r.search_end_ms - r.search_start_ms, left=r.search_start_ms, 
                 height=bar_height, color=colors['search'], alpha=0.8, label='Search')
        # Resize bar (shrink)
        ax3.barh(y_pos - 0.4, r.resize_end_ms - r.resize_start_ms, left=r.resize_start_ms,
                 height=bar_height, color=colors['resize'], alpha=0.8, label='Shrink')
        # Pure resize portion
        ax3.barh(y_pos - 0.4, r.pure_resize_latency_ms, left=r.resize_end_ms - r.pure_resize_latency_ms,
                 height=bar_height, color=colors['resize'], alpha=1.0, edgecolor='black', linewidth=2)
        y_pos -= 1.2
        
    if rep_expand:
        r = rep_expand[0]
        # Search bar
        ax3.barh(y_pos, r.search_end_ms - r.search_start_ms, left=r.search_start_ms, 
                 height=bar_height, color=colors['search'], alpha=0.8)
        # Resize bar (expand)
        ax3.barh(y_pos - 0.4, r.resize_end_ms - r.resize_start_ms, left=r.resize_start_ms,
                 height=bar_height, color=colors['expand'], alpha=0.8, label='Expand')
        # Pure resize portion
        ax3.barh(y_pos - 0.4, r.pure_resize_latency_ms, left=r.resize_end_ms - r.pure_resize_latency_ms,
                 height=bar_height, color=colors['expand'], alpha=1.0, edgecolor='black', linewidth=2)
    
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('')
    ax3.set_title(f'Timeline: Concurrent Search+Resize (nq={mid_batch}, delta={int(mid_delta/(1024**2))} MB)\n'
                  f'[Dark portion = Pure Resize, Light = Blocked waiting for Search]')
    ax3.set_yticks([0, -0.4, -1.2, -1.6])
    ax3.set_yticklabels(['Shrink+Search', 'Shrink Resize', 'Expand+Search', 'Expand Resize'])
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(axis='x', alpha=0.3)
    
    # Plot 4: Resize latency by delta size (showing pure vs total)
    ax4 = axes[1, 1]
    
    # Get pure resize latencies and total resize latencies by delta
    expand_pure_by_delta = []
    expand_total_by_delta = []
    shrink_pure_by_delta = []
    shrink_total_by_delta = []
    
    for d in delta_sizes:
        expand_r = [r for r in expand_results if r.resize_delta_bytes == d]
        shrink_r = [r for r in shrink_results if r.resize_delta_bytes == d]
        
        if expand_r:
            expand_pure_by_delta.append(expand_r[0].pure_resize_latency_ms)
            expand_total_by_delta.append(sum(r.resize_latency_ms for r in expand_r) / len(expand_r))
        else:
            expand_pure_by_delta.append(0)
            expand_total_by_delta.append(0)
            
        if shrink_r:
            shrink_pure_by_delta.append(shrink_r[0].pure_resize_latency_ms)
            shrink_total_by_delta.append(sum(r.resize_latency_ms for r in shrink_r) / len(shrink_r))
        else:
            shrink_pure_by_delta.append(0)
            shrink_total_by_delta.append(0)
    
    x = np.arange(len(delta_sizes_mb))
    width = 0.2
    
    ax4.bar(x - 1.5*width, expand_pure_by_delta, width, label='Expand Pure', color='#27ae60', alpha=0.9)
    ax4.bar(x - 0.5*width, expand_total_by_delta, width, label='Expand +Search', color='#2ecc71', alpha=0.5, hatch='//')
    ax4.bar(x + 0.5*width, shrink_pure_by_delta, width, label='Shrink Pure', color='#c0392b', alpha=0.9)
    ax4.bar(x + 1.5*width, shrink_total_by_delta, width, label='Shrink +Search', color='#e74c3c', alpha=0.5, hatch='\\\\')
    
    ax4.set_xlabel('Delta Size (MB)')
    ax4.set_ylabel('Resize Latency (ms)')
    ax4.set_title('Resize Latency: Pure (no search) vs Concurrent (avg across batch sizes)')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'{d:.0f}' for d in delta_sizes_mb])
    ax4.legend(loc='upper left', fontsize=8)
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved latency breakdown plot to: {output_path}")


# ============================================================================
# Main
# ============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GPU Memory Pool Resize Overhead Benchmark"
    )
    
    group = parser.add_argument_group("data and index")
    group.add_argument("--nb", type=int, default=100_000, help="Number of database vectors")
    group.add_argument("--dim", type=int, default=128, help="Vector dimension")
    group.add_argument("--nlist", type=int, default=100, help="Number of IVF clusters")
    group.add_argument("--nprobe", type=int, default=10, help="Clusters to probe per query")
    group.add_argument("--k", type=int, default=10, help="k nearest neighbors")
    
    group = parser.add_argument_group("GPU memory")
    group.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    group.add_argument("--base-pool-mb", type=int, default=256, 
                       help="Base pool size in MB")
    
    group = parser.add_argument_group("benchmark parameters")
    group.add_argument("--nruns", type=int, default=5, help="Number of runs per test")
    group.add_argument("--batch-sizes", type=str, default="100,500,1000,5000,10000",
                       help="Comma-separated query batch sizes")
    group.add_argument("--delta-sizes-mb", type=str, default="16,32,64,128,256",
                       help="Comma-separated resize delta sizes in MB")
    
    group = parser.add_argument_group("output")
    group.add_argument("--output-dir", type=str, default=".",
                       help="Output directory for plots")
    group.add_argument("--prefix", type=str, default="resize_bench",
                       help="Prefix for output files")
    
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    
    # Check GPU availability
    ngpu = faiss.get_num_gpus()
    if ngpu < 1:
        raise RuntimeError("No GPU available")
    if args.gpu >= ngpu:
        raise RuntimeError(f"GPU {args.gpu} not available (only {ngpu} GPUs found)")
    
    print(f"=" * 60)
    print("GPU Memory Pool Resize Overhead Benchmark")
    print(f"=" * 60)
    
    # Parse parameters
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
    delta_sizes_mb = [int(x.strip()) for x in args.delta_sizes_mb.split(",")]
    
    cfg = BenchmarkConfig(
        device_id=args.gpu,
        dim=args.dim,
        nb=args.nb,
        nlist=args.nlist,
        nprobe=args.nprobe,
        k=args.k,
        base_pool_size_mb=args.base_pool_mb,
        nruns=args.nruns,
        output_dir=args.output_dir,
    )
    
    print(f"\nConfiguration:")
    print(f"  GPU: {cfg.device_id}")
    print(f"  Database: {cfg.nb} vectors, dim={cfg.dim}")
    print(f"  Index: IVF{cfg.nlist}, nprobe={cfg.nprobe}")
    print(f"  Base pool size: {cfg.base_pool_size_mb} MB")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Delta sizes (MB): {delta_sizes_mb}")
    print(f"  Runs per test: {cfg.nruns}")
    
    # Create index and resources
    res, gpu_index, xb = create_gpu_index_and_resources(cfg)
    
    # Generate query vectors
    max_nq = max(batch_sizes)
    xq = np.random.randn(max_nq, cfg.dim).astype(np.float32) * 0.1
    
    # Open pool controller
    ctrl = GpuPoolController(cfg.device_id)
    
    try:
        # Warmup
        warmup_operations(gpu_index, ctrl, xq, cfg.k, nruns=2)
        
        # Benchmark 1: Pure resize operations
        print(f"\n{'=' * 60}")
        print("Benchmark 1: Pure Resize Operations")
        print(f"{'=' * 60}")
        pure_resize_results = bench_pure_resize(cfg, ctrl, delta_sizes_mb)
        
        # Benchmark 2: Search baseline
        print(f"\n{'=' * 60}")
        print("Benchmark 2: Search Baseline (no resize)")
        print(f"{'=' * 60}")
        search_baseline = bench_search_baseline(cfg, gpu_index, xq, batch_sizes)
        
        # Benchmark 3: Concurrent search + resize
        print(f"\n{'=' * 60}")
        print("Benchmark 3: Concurrent Search + Resize")
        print(f"{'=' * 60}")
        concurrent_results = bench_concurrent_search_resize(
            cfg, gpu_index, ctrl, xq, batch_sizes, delta_sizes_mb,
            pure_resize_results=pure_resize_results,
        )
        
        # Generate plots
        print(f"\n{'=' * 60}")
        print("Generating Plots")
        print(f"{'=' * 60}")
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        plot_pure_resize_results(
            pure_resize_results,
            os.path.join(args.output_dir, f"{args.prefix}_pure_resize.png"),
        )
        
        plot_concurrent_results(
            concurrent_results,
            search_baseline,
            os.path.join(args.output_dir, f"{args.prefix}_concurrent.png"),
        )
        
        plot_resize_latency_breakdown(
            concurrent_results,
            os.path.join(args.output_dir, f"{args.prefix}_breakdown.png"),
        )
        
        # Print summary
        print(f"\n{'=' * 60}")
        print("Summary")
        print(f"{'=' * 60}")
        
        print("\nPure Resize Latency (ms):")
        print(f"  {'Delta (MB)':<12} {'Expand':<15} {'Shrink':<15}")
        for i, delta_mb in enumerate(delta_sizes_mb):
            expand_lat = pure_resize_results["expand"][i].latency_ms
            shrink_lat = pure_resize_results["shrink"][i].latency_ms
            print(f"  {delta_mb:<12} {expand_lat:<15.2f} {shrink_lat:<15.2f}")
        
        print("\nSearch Baseline (ms):")
        for nq, lat in search_baseline.items():
            print(f"  nq={nq}: {lat:.2f} ms")
        
        print(f"\nOutput files saved to: {args.output_dir}/")
        print(f"  - {args.prefix}_pure_resize.png")
        print(f"  - {args.prefix}_concurrent.png")
        print(f"  - {args.prefix}_breakdown.png")
        
    finally:
        ctrl.close()
        del gpu_index
        del res
        gc.collect()


if __name__ == "__main__":
    main()
