# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
IVF IPC Glue - Low-level IPC protocol for GpuIndexIVFFlat.

This module provides the low-level shared memory IPC protocol for
communicating with GpuIndexIVFFlat instances. It mirrors the C++
IpcCommand structure defined in GpuIndexIVFFlat.h.

Protocol:
1. Create transient shared memory segment with unique name
2. Write IpcCommand structure (32 bytes):
   - magic (uint32): 0x4956464C ("IVFL")
   - version (uint32): 1
   - state (uint32): 0=Idle, 1=Pending, 2=Done, 3=Error
   - opcode (uint32): 1=Evict, 2=Load
   - listId (int64): Target IVF list ID
   - result (int64): Return value (bytes evicted/loaded, or -1 for error)
3. Call index.processSharedMemoryCommand(shm_name)
4. Poll until state != Pending (typically becomes Done or Error)
5. Read result and cleanup
"""

import ctypes
import mmap
import os
import struct
import time
import uuid
from typing import Optional, Tuple

# IPC protocol constants (must match GpuIndexIVFFlat.h)
KIpcMagic = 0x4956464C  # "IVFL"
KIpcVersion = 1

# State constants
KIpcStateIdle = 0
KIpcStatePending = 1
KIpcStateDone = 2
KIpcStateError = 3

# Opcode constants
KIpcOpEvict = 1
KIpcOpLoad = 2

# Struct format: magic, version, state, opcode, listId, result
# All fields are native endian, no padding (packed)
IPC_STRUCT_FORMAT = "=IIIIqq"
IPC_STRUCT_SIZE = struct.calcsize(IPC_STRUCT_FORMAT)  # Should be 32 bytes


def _get_libc() -> ctypes.CDLL:
    """Get libc handle for shm operations."""
    return ctypes.CDLL("libc.so.6", use_errno=True)


def _shm_open(name: str, size: int) -> int:
    """
    Create/open shared memory segment.
    
    Args:
        name: Shared memory name (must start with "/")
        size: Size in bytes
        
    Returns:
        File descriptor
        
    Raises:
        OSError: If shm_open or ftruncate fails
        ValueError: If name or size is invalid
    """
    if not isinstance(name, str) or not name.startswith("/"):
        raise ValueError("shm name must be a string starting with '/'")
    if size <= 0:
        raise ValueError("size must be > 0")
    
    libc = _get_libc()
    
    shm_open_fn = libc.shm_open
    shm_open_fn.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    shm_open_fn.restype = ctypes.c_int
    
    fd = shm_open_fn(name.encode("utf-8"), os.O_CREAT | os.O_RDWR, 0o666)
    if fd < 0:
        err = ctypes.get_errno()
        raise OSError(err, f"shm_open failed for {name!r}")
    
    ftruncate_fn = libc.ftruncate
    ftruncate_fn.argtypes = [ctypes.c_int, ctypes.c_long]
    ftruncate_fn.restype = ctypes.c_int
    
    if ftruncate_fn(fd, size) != 0:
        err = ctypes.get_errno()
        os.close(fd)
        raise OSError(err, f"ftruncate failed for {name!r}")
    
    return fd


def _shm_unlink(name: str) -> None:
    """Remove shared memory segment."""
    libc = _get_libc()
    shm_unlink_fn = libc.shm_unlink
    shm_unlink_fn.argtypes = [ctypes.c_char_p]
    shm_unlink_fn.restype = ctypes.c_int
    shm_unlink_fn(name.encode("utf-8"))


def _pack_ipc_command(
    magic: int = KIpcMagic,
    version: int = KIpcVersion,
    state: int = KIpcStateIdle,
    opcode: int = 0,
    list_id: int = 0,
    result: int = 0,
) -> bytes:
    """Pack IPC command structure to bytes."""
    return struct.pack(
        IPC_STRUCT_FORMAT,
        magic,
        version,
        state,
        opcode,
        int(list_id),
        int(result),
    )


def _unpack_ipc_command(buf: bytes) -> dict:
    """Unpack IPC command structure from bytes."""
    if len(buf) != IPC_STRUCT_SIZE:
        raise ValueError(f"Buffer size mismatch: expected {IPC_STRUCT_SIZE}, got {len(buf)}")
    
    magic, version, state, opcode, list_id, result = struct.unpack(IPC_STRUCT_FORMAT, buf)
    return {
        "magic": magic,
        "version": version,
        "state": state,
        "opcode": opcode,
        "list_id": list_id,
        "result": result,
    }


def _generate_shm_name() -> str:
    """Generate unique shared memory name."""
    return f"/faissctl_ivf_ipc_{uuid.uuid4().hex}"


class IvfIpcError(Exception):
    """Base exception for IVF IPC errors."""
    pass


class IvfIpcTimeoutError(IvfIpcError):
    """Raised when IPC command times out."""
    pass


class IvfIpcProtocolError(IvfIpcError):
    """Raised when IPC protocol is violated."""
    pass


def send_ivf_ipc_command(
    index,
    opcode: int,
    list_id: int,
    timeout_ms: int = 5000,
    poll_interval_ms: int = 10,
) -> Tuple[bool, int]:
    """
    Send IPC command to GpuIndexIVFFlat via shared memory.
    
    This is the core IPC function that communicates with the C++
    processSharedMemoryCommand method.
    
    Args:
        index: GpuIndexIVFFlat instance (must have processSharedMemoryCommand method)
        opcode: Command opcode (KIpcOpEvict=1 or KIpcOpLoad=2)
        list_id: Target IVF list ID
        timeout_ms: Maximum time to wait for completion
        poll_interval_ms: Polling interval
        
    Returns:
        Tuple of (handled, result):
        - handled: True if command was processed by index
        - result: Bytes evicted/loaded, or -1 if error
        
    Raises:
        IvfIpcTimeoutError: If command times out
        IvfIpcProtocolError: If protocol error occurs
        AttributeError: If index doesn't have processSharedMemoryCommand method
    """
    if not hasattr(index, "processSharedMemoryCommand"):
        raise AttributeError(
            "Index must have processSharedMemoryCommand method. "
            "Ensure you're using GpuIndexIVFFlat or a compatible GPU index."
        )
    
    shm_name = _generate_shm_name()
    fd = -1
    mm = None
    
    try:
        # Create shared memory segment
        fd = _shm_open(shm_name, IPC_STRUCT_SIZE)
        mm = mmap.mmap(fd, IPC_STRUCT_SIZE, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        
        # Write command with Pending state
        mm.seek(0)
        mm.write(_pack_ipc_command(
            state=KIpcStatePending,
            opcode=opcode,
            list_id=list_id,
        ))
        mm.flush()
        
        # Send command to index
        handled = bool(index.processSharedMemoryCommand(shm_name))
        
        if not handled:
            # Command not handled (e.g., wrong magic/version or not in Pending state)
            return False, -1
        
        # Poll for completion
        start_time = time.time()
        while True:
            mm.seek(0)
            buf = mm.read(IPC_STRUCT_SIZE)
            cmd = _unpack_ipc_command(buf)
            
            state = cmd["state"]
            
            if state == KIpcStateDone:
                # Success
                return True, cmd["result"]
            elif state == KIpcStateError:
                # Error during processing
                return True, -1
            elif state == KIpcStateIdle:
                # Should not happen if handled was True, but handle gracefully
                raise IvfIpcProtocolError("Command state reverted to Idle unexpectedly")
            elif state != KIpcStatePending:
                # Unknown state
                raise IvfIpcProtocolError(f"Unknown IPC state: {state}")
            
            # Check timeout
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms >= timeout_ms:
                raise IvfIpcTimeoutError(
                    f"IPC command timed out after {timeout_ms}ms. "
                    f"The index may be unresponsive."
                )
            
            # Wait before next poll
            time.sleep(poll_interval_ms / 1000.0)
    
    finally:
        # Cleanup
        if mm is not None:
            mm.close()
        if fd >= 0:
            os.close(fd)
        _shm_unlink(shm_name)


def evict_list(index, list_id: int, timeout_ms: int = 5000) -> Tuple[bool, int]:
    """
    Evict an IVF list from GPU to CPU.
    
    Args:
        index: GpuIndexIVFFlat instance
        list_id: List ID to evict
        timeout_ms: Timeout in milliseconds
        
    Returns:
        Tuple of (success, bytes_evicted)
        - success: True if eviction succeeded
        - bytes_evicted: Number of bytes evicted, or -1 on error
    """
    handled, result = send_ivf_ipc_command(
        index, opcode=KIpcOpEvict, list_id=list_id, timeout_ms=timeout_ms
    )
    return handled and result >= 0, result


def load_list(index, list_id: int, timeout_ms: int = 5000) -> Tuple[bool, int]:
    """
    Load an IVF list from CPU to GPU.
    
    Args:
        index: GpuIndexIVFFlat instance
        list_id: List ID to load
        timeout_ms: Timeout in milliseconds
        
    Returns:
        Tuple of (success, bytes_loaded)
        - success: True if load succeeded
        - bytes_loaded: Number of bytes loaded, or -1 on error
    """
    handled, result = send_ivf_ipc_command(
        index, opcode=KIpcOpLoad, list_id=list_id, timeout_ms=timeout_ms
    )
    return handled and result >= 0, result


def find_owning_shard(index_shards, list_id: int):
    """
    Find which shard in a multi-GPU index owns a specific list.
    
    Args:
        index_shards: IndexShardsIVF instance (has count() and at() methods)
        list_id: List ID to find
        
    Returns:
        Shard index that owns the list, or None if not found
        
    Raises:
        AttributeError: If index_shards doesn't have required methods
    """
    if not (hasattr(index_shards, "count") and hasattr(index_shards, "at")):
        raise AttributeError("Index must have count() and at() methods")
    
    import faiss
    
    for s in range(index_shards.count()):
        shard = faiss.downcast_index(index_shards.at(s))
        if hasattr(shard, "isListOnGpu"):
            # List could be on this shard (or was on this shard)
            return s
        elif hasattr(shard, "processSharedMemoryCommand"):
            # Assume this shard might have the list
            return s
    
    return None
