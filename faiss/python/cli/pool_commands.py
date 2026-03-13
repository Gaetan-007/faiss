# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Pool management commands for faissctl.

Provides commands for managing GPU memory pools via IPC:
- pool list: List all GPU pools
- pool stats: Show pool statistics
- pool expand: Expand pool size
- pool shrink: Shrink pool size
"""

import glob
import sys
from typing import List, Optional

from .utils import (
    clr,
    format_percentage,
    format_size,
    parse_size,
    print_error,
    print_success,
    print_warning,
    PoolNotFoundError,
)


def get_pool_controller_safe(device_id: int):
    """
    Safely get pool controller for device.
    
    Args:
        device_id: GPU device ID
        
    Returns:
        GpuPoolController instance
        
    Raises:
        PoolNotFoundError: If pool doesn't exist or IPC not available
    """
    try:
        # Import here to avoid heavy import at module load time
        from faiss.gpu_pool_controller import get_pool_controller
        
        ctrl = get_pool_controller(device_id)
        if ctrl is None:
            raise PoolNotFoundError(
                f"GPU {device_id} pool not found. "
                f"Make sure the pool is initialized with IPC enabled."
            )
        return ctrl
    except ImportError as e:
        raise PoolNotFoundError(f"Failed to import GpuPoolController: {e}")
    except Exception as e:
        raise PoolNotFoundError(f"Failed to connect to GPU {device_id} pool: {e}")


def detect_gpu_pools() -> List[int]:
    """
    Detect all available GPU pools by scanning /dev/shm.
    
    Returns:
        List of device IDs with available pools
    """
    pools = []
    pool_paths = glob.glob("/dev/shm/faiss_gpu_pool_ctrl_*")
    
    for pool_path in sorted(pool_paths):
        try:
            # Extract device ID from path like /dev/shm/faiss_gpu_pool_ctrl_0
            dev_id_str = pool_path.split("_")[-1]
            dev_id = int(dev_id_str)
            pools.append(dev_id)
        except (ValueError, IndexError):
            continue
    
    return pools


def cmd_pool_list(json_output: bool = False) -> int:
    """
    List all GPU memory pools.
    
    Args:
        json_output: If True, output as JSON
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    pools = detect_gpu_pools()
    
    if not pools:
        print_warning("No GPU memory pools found.")
        print("Make sure Faiss GPU pools are initialized with IPC enabled.")
        return 0
    
    if json_output:
        import json
        pool_data = []
        for dev_id in pools:
            try:
                ctrl = get_pool_controller_safe(dev_id)
                stats = ctrl.get_stats()
                pool_data.append({
                    "device_id": dev_id,
                    "total_bytes": stats["total_bytes"],
                    "used_bytes": stats["used_bytes"],
                    "available_bytes": stats["available_bytes"],
                    "utilization": stats["utilization"],
                })
            except PoolNotFoundError as e:
                pool_data.append({
                    "device_id": dev_id,
                    "error": str(e),
                })
        print(json.dumps(pool_data, indent=2))
        return 0
    
    # Print table header
    header = f"{'Device':<8} {'Total':<12} {'Used':<12} {'Available':<12} {'Util':<10}"
    print(clr(header, 'cyan', bold=True))
    print("-" * 60)
    
    for dev_id in pools:
        try:
            ctrl = get_pool_controller_safe(dev_id)
            stats = ctrl.get_stats()
            
            total = format_size(stats["total_bytes"])
            used = format_size(stats["used_bytes"])
            available = format_size(stats["available_bytes"])
            util = format_percentage(stats["used_bytes"], stats["total_bytes"])
            
            print(f"{dev_id:<8} {total:<12} {used:<12} {available:<12} {util:<10}")
        except PoolNotFoundError as e:
            print(f"{dev_id:<8} {clr(str(e), 'red')}")
    
    return 0


def cmd_pool_stats(device_id: int, json_output: bool = False) -> int:
    """
    Show detailed statistics for a specific GPU pool.
    
    Args:
        device_id: GPU device ID
        json_output: If True, output as JSON
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        ctrl = get_pool_controller_safe(device_id)
        stats = ctrl.get_stats()
    except PoolNotFoundError as e:
        print_error(str(e))
        return 1
    
    if json_output:
        import json
        print(json.dumps(stats, indent=2))
        return 0
    
    # Print formatted stats
    print(clr(f"GPU {device_id} Memory Pool Statistics", 'cyan', bold=True))
    print("=" * 50)
    print(f"  Total Size:     {format_size(stats['total_bytes'])}")
    print(f"  Used:           {format_size(stats['used_bytes'])}")
    print(f"  Available:      {format_size(stats['available_bytes'])}")
    print(f"  Utilization:    {format_percentage(stats['used_bytes'], stats['total_bytes'])}")
    print()
    
    return 0


def cmd_pool_expand(device_id: int, target_size: str, timeout_ms: int = 5000) -> int:
    """
    Expand GPU memory pool to at least the target size.
    
    Args:
        device_id: GPU device ID
        target_size: Target size (e.g., "2G", "512M")
        timeout_ms: Timeout in milliseconds
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        target_bytes = parse_size(target_size)
    except ValueError as e:
        print_error(f"Invalid size '{target_size}': {e}")
        return 1
    
    try:
        ctrl = get_pool_controller_safe(device_id)
    except PoolNotFoundError as e:
        print_error(str(e))
        return 1
    
    print(f"Expanding GPU {device_id} pool to {format_size(target_bytes)}...")
    
    try:
        result = ctrl.expand(target_bytes, timeout_ms=timeout_ms)
        
        status = result.get("status", -1)
        actual_size = result.get("actual_size", 0)
        error = result.get("error", "")
        
        if status == 1:  # SUCCESS
            print_success(
                f"Pool expanded successfully. "
                f"Current size: {format_size(actual_size)}"
            )
            return 0
        elif status == 2:  # FAILED
            print_error(f"Expansion failed: {error}")
            return 1
        else:
            print_error(f"Unexpected status: {status}")
            return 1
    except Exception as e:
        print_error(f"Expansion failed: {e}")
        return 1


def cmd_pool_shrink(device_id: int, target_size: str, timeout_ms: int = 5000) -> int:
    """
    Shrink GPU memory pool to at most the target size.
    
    Args:
        device_id: GPU device ID
        target_size: Target size (e.g., "1G", "256M")
        timeout_ms: Timeout in milliseconds
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        target_bytes = parse_size(target_size)
    except ValueError as e:
        print_error(f"Invalid size '{target_size}': {e}")
        return 1
    
    try:
        ctrl = get_pool_controller_safe(device_id)
    except PoolNotFoundError as e:
        print_error(str(e))
        return 1
    
    # Get current stats to warn if target is below used
    stats = ctrl.get_stats()
    if target_bytes < stats["used_bytes"]:
        print_warning(
            f"Target size {format_size(target_bytes)} is less than "
            f"used memory {format_size(stats['used_bytes'])}. "
            f"Pool can only shrink to fit actual usage."
        )
    
    print(f"Shrinking GPU {device_id} pool to {format_size(target_bytes)}...")
    
    try:
        result = ctrl.shrink(target_bytes, timeout_ms=timeout_ms)
        
        status = result.get("status", -1)
        actual_size = result.get("actual_size", 0)
        error = result.get("error", "")
        
        if status == 1:  # SUCCESS
            print_success(
                f"Pool shrunk successfully. "
                f"Current size: {format_size(actual_size)}"
            )
            return 0
        elif status == 3:  # PARTIAL
            print_warning(
                f"Pool partially shrunk to {format_size(actual_size)}. "
                f"Some chunks could not be released (may be in use)."
            )
            return 0
        elif status == 2:  # FAILED
            print_error(f"Shrink failed: {error}")
            return 1
        else:
            print_error(f"Unexpected status: {status}")
            return 1
    except Exception as e:
        print_error(f"Shrink failed: {e}")
        return 1
