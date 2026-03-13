# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
IVF list management commands for faissctl.

Provides commands for managing IVF lists on GPU indices:
- ivf status: Show evicted lists
- ivf evict: Evict list to CPU
- ivf load: Load list to GPU
"""

import os
import sys
from typing import List, Optional, Tuple

import faiss

from .ipc_glue import evict_list, load_list
from .utils import clr, format_size, print_error, print_success, print_warning


def load_index(index_path: str) -> faiss.Index:
    """
    Load Faiss index from file.
    
    Args:
        index_path: Path to index file
        
    Returns:
        Loaded index
        
    Raises:
        FileNotFoundError: If file doesn't exist
        RuntimeError: If loading fails
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file not found: {index_path}")
    
    try:
        index = faiss.read_index(index_path)
        return index
    except Exception as e:
        raise RuntimeError(f"Failed to load index: {e}")


def get_index_nlist(index) -> Optional[int]:
    """Get nlist from index if available."""
    if hasattr(index, "nlist"):
        return index.nlist
    return None


def is_gpu_ivf_index(index) -> bool:
    """Check if index is a GPU IVF index with IPC support."""
    return hasattr(index, "processSharedMemoryCommand")


def is_multi_gpu_index(index) -> bool:
    """Check if index is a multi-GPU sharded index."""
    return hasattr(index, "count") and hasattr(index, "at")


def find_shard_with_list(index_shards, list_id: int) -> Optional[Tuple[int, object]]:
    """
    Find which shard owns a specific list in multi-GPU index.
    
    Args:
        index_shards: Multi-GPU index (IndexShardsIVF)
        list_id: List ID to find
        
    Returns:
        Tuple of (shard_index, shard) or None
    """
    for s in range(index_shards.count()):
        shard = faiss.downcast_index(index_shards.at(s))
        if hasattr(shard, "isListOnGpu"):
            # Check if this shard has the list
            try:
                if shard.isListOnGpu(list_id):
                    return s, shard
            except Exception:
                pass
        elif hasattr(shard, "processSharedMemoryCommand"):
            # Assume this shard might handle the list
            return s, shard
    return None


def get_evicted_lists(index) -> List[int]:
    """
    Get list of evicted (on CPU) IVF lists.
    
    Args:
        index: GPU IVF index
        
    Returns:
        List of list IDs currently on CPU
    """
    if hasattr(index, "getEvictedLists"):
        return index.getEvictedLists()
    return []


def cmd_ivf_status(index_path: str, json_output: bool = False) -> int:
    """
    Show status of IVF lists (which are on GPU vs CPU).
    
    Args:
        index_path: Path to index file
        json_output: If True, output as JSON
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        index = load_index(index_path)
    except (FileNotFoundError, RuntimeError) as e:
        print_error(str(e))
        return 1
    
    # Check if GPU IVF index
    if not is_gpu_ivf_index(index):
        if is_multi_gpu_index(index):
            # Multi-GPU index - check shards
            total_evicted = 0
            total_lists = 0
            shard_info = []
            
            for s in range(index.count()):
                shard = faiss.downcast_index(index.at(s))
                if hasattr(shard, "getEvictedLists"):
                    evicted = shard.getEvictedLists()
                    nlist = getattr(shard, "nlist", len(evicted))
                    total_evicted += len(evicted)
                    total_lists += nlist
                    shard_info.append({
                        "shard": s,
                        "evicted": len(evicted),
                        "total": nlist,
                    })
            
            if json_output:
                import json
                print(json.dumps(shard_info, indent=2))
                return 0
            
            print(clr(f"IVF List Status: {index_path}", 'cyan', bold=True))
            print(f"Total: {total_lists - total_evicted} on GPU, {total_evicted} on CPU")
            print()
            for info in shard_info:
                pct = (info["evicted"] / info["total"] * 100) if info["total"] > 0 else 0
                print(f"  Shard {info['shard']}: {info['total'] - info['evicted']}/{info['total']} on GPU ({pct:.1f}% evicted)")
            return 0
        else:
            print_error("Index is not a GPU IVF index with IPC support")
            return 1
    
    # Single GPU index
    evicted = get_evicted_lists(index)
    nlist = get_index_nlist(index)
    
    if nlist is None:
        print_error("Cannot determine nlist for index")
        return 1
    
    if json_output:
        import json
        result = {
            "index_path": index_path,
            "nlist": nlist,
            "on_gpu": nlist - len(evicted),
            "on_cpu": len(evicted),
            "evicted_list_ids": evicted,
        }
        print(json.dumps(result, indent=2))
        return 0
    
    # Print formatted output
    print(clr(f"IVF List Status: {index_path}", 'cyan', bold=True))
    print(f"Total lists: {nlist}")
    print(f"  On GPU: {nlist - len(evicted)}")
    print(f"  On CPU (evicted): {len(evicted)}")
    
    if evicted:
        print()
        print("Evicted list IDs:")
        for list_id in sorted(evicted):
            print(f"  - {list_id}")
    
    return 0


def cmd_ivf_evict(index_path: str, list_id: int, timeout_ms: int = 5000) -> int:
    """
    Evict an IVF list from GPU to CPU.
    
    Args:
        index_path: Path to index file
        list_id: List ID to evict
        timeout_ms: Timeout in milliseconds
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        index = load_index(index_path)
    except (FileNotFoundError, RuntimeError) as e:
        print_error(str(e))
        return 1
    
    if is_multi_gpu_index(index):
        # Find owning shard
        result = find_shard_with_list(index, list_id)
        if result is None:
            print_error(f"List {list_id} not found in any shard")
            return 1
        shard_idx, shard = result
        target_index = shard
        print(f"Found list {list_id} in shard {shard_idx}")
    elif is_gpu_ivf_index(index):
        target_index = index
    else:
        print_error("Index is not a GPU IVF index with IPC support")
        return 1
    
    # Check if list is already evicted
    if hasattr(target_index, "isListOnGpu"):
        if not target_index.isListOnGpu(list_id):
            print_warning(f"List {list_id} is already on CPU (evicted)")
            return 0
    
    # Perform eviction
    print(f"Evicting list {list_id}...")
    try:
        success, bytes_evicted = evict_list(target_index, list_id, timeout_ms)
        if success:
            print_success(f"List {list_id} evicted successfully ({format_size(bytes_evicted)})")
            return 0
        else:
            print_error(f"Failed to evict list {list_id}")
            return 1
    except Exception as e:
        print_error(f"Eviction failed: {e}")
        return 1


def cmd_ivf_load(index_path: str, list_id: int, timeout_ms: int = 5000) -> int:
    """
    Load an IVF list from CPU to GPU.
    
    Args:
        index_path: Path to index file
        list_id: List ID to load
        timeout_ms: Timeout in milliseconds
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        index = load_index(index_path)
    except (FileNotFoundError, RuntimeError) as e:
        print_error(str(e))
        return 1
    
    if is_multi_gpu_index(index):
        # Find owning shard
        result = find_shard_with_list(index, list_id)
        if result is None:
            print_error(f"List {list_id} not found in any shard")
            return 1
        shard_idx, shard = result
        target_index = shard
        print(f"Found list {list_id} in shard {shard_idx}")
    elif is_gpu_ivf_index(index):
        target_index = index
    else:
        print_error("Index is not a GPU IVF index with IPC support")
        return 1
    
    # Check if list is already on GPU
    if hasattr(target_index, "isListOnGpu"):
        if target_index.isListOnGpu(list_id):
            print_warning(f"List {list_id} is already on GPU")
            return 0
    
    # Perform load
    print(f"Loading list {list_id}...")
    try:
        success, bytes_loaded = load_list(target_index, list_id, timeout_ms)
        if success:
            print_success(f"List {list_id} loaded successfully ({format_size(bytes_loaded)})")
            return 0
        else:
            print_error(f"Failed to load list {list_id}")
            return 1
    except Exception as e:
        print_error(f"Load failed: {e}")
        return 1
