# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
GPU Memory Pool Controller

This module provides a Python client for controlling the PreallocMemoryPool
via shared memory IPC. It allows external processes to dynamically expand
or shrink the GPU memory pool at runtime.

Usage:
    from faiss.gpu_pool_controller import GpuPoolController

    # Connect to the pool for GPU device 0
    ctrl = GpuPoolController(device_id=0)

    # Query current state
    state = ctrl.query()
    print(f"Total: {state['actual_size']}, Available: {state['available']}")

    # Expand to 2GB
    result = ctrl.expand(2 * 1024 * 1024 * 1024)

    # Shrink to 1GB
    result = ctrl.shrink(1 * 1024 * 1024 * 1024)
"""

import mmap
import os
import struct
import time
from enum import IntEnum
from typing import Optional

__all__ = [
    "GpuPoolController",
    "ResizeCommand",
    "ResizeStatus",
    "get_pool_controller",
]


class ResizeCommand(IntEnum):
    """Commands that can be sent to the memory pool."""

    NOP = 0
    EXPAND_TO = 1  # Expand pool to at least targetSizeBytes
    SHRINK_TO = 2  # Shrink pool down to targetSizeBytes
    QUERY = 3  # Query current pool state
    EXPAND_BY = 4  # Expand pool by deltaBytes


class ResizeStatus(IntEnum):
    """Status codes returned by the memory pool."""

    PENDING = 0
    SUCCESS = 1
    FAILED = 2
    PARTIAL = 3  # For shrink when we cannot fully reach the target


class GpuPoolController:
    """
    Controller for the PreallocMemoryPool via shared memory IPC.

    This class provides methods to dynamically resize the GPU memory pool
    from an external process. The pool must be initialized with IPC enabled
    (which is the default when using setDeviceMemoryReservation).

    Attributes:
        device_id: The GPU device ID this controller is connected to.

    Example:
        >>> ctrl = GpuPoolController(device_id=0)
        >>> result = ctrl.query()
        >>> print(f"Pool size: {result['actual_size']} bytes")

        >>> # Expand to 4GB
        >>> result = ctrl.expand(4 * 1024**3)
        >>> if result['status'] == ResizeStatus.SUCCESS:
        ...     print("Expansion successful")

        >>> # Shrink to 2GB
        >>> result = ctrl.shrink(2 * 1024**3)
    """

    # Shared memory control block format (must match ShmControlBlock in C++)
    # Memory layout (total 312 bytes):
    #   Offset  Size  Field
    #   ------  ----  -----
    #      0      4   version (uint32_t)
    #      4      4   command (uint32_t, ResizeCommand)
    #      8      4   status (uint32_t, ResizeStatus)
    #     12      4   reserved (uint32_t, padding)
    #     16      4   deviceId (int32_t)
    #     20      4   reserved2 (int32_t, padding)
    #     24      8   targetSizeBytes (int64_t)
    #     32      8   deltaBytes (int64_t)
    #     40      8   actualSizeBytes (int64_t)
    #     48      8   availableBytes (int64_t)
    #     56    256   errorMsg (char[256])
    CTRL_FORMAT = "=IIIIiiqqqq256s"
    CTRL_SIZE = struct.calcsize(CTRL_FORMAT)  # Should be 312 bytes

    # Field indices in unpacked tuple
    _IDX_VERSION = 0
    _IDX_COMMAND = 1
    _IDX_STATUS = 2
    _IDX_RESERVED = 3
    _IDX_DEVICE_ID = 4
    _IDX_RESERVED2 = 5
    _IDX_TARGET_SIZE = 6
    _IDX_DELTA = 7
    _IDX_ACTUAL_SIZE = 8
    _IDX_AVAILABLE = 9
    _IDX_ERROR_MSG = 10

    def __init__(self, device_id: int):
        """
        Initialize the controller for the specified GPU device.

        Args:
            device_id: The GPU device ID to connect to.

        Raises:
            FileNotFoundError: If the shared memory segment does not exist
                (pool not initialized or IPC disabled).
            OSError: If there's an error opening or mapping the shared memory.
        """
        # Initialize _shm early so __del__ can safely call close() even if
        # __init__ fails before _shm is assigned
        self._shm = None

        if device_id < 0:
            raise ValueError("device_id must be non-negative")
        self.device_id = device_id
        self._shm_name = f"/faiss_gpu_pool_ctrl_{device_id}"
        self._shm_path = f"/dev/shm{self._shm_name}"

        if not os.path.exists(self._shm_path):
            raise FileNotFoundError(
                f"Shared memory segment {self._shm_name} not found. "
                f"Make sure the PreallocMemoryPool is initialized with IPC enabled "
                f"for device {device_id}."
            )

        shm_stat = os.stat(self._shm_path)
        if shm_stat.st_size < self.CTRL_SIZE:
            raise OSError(
                f"Shared memory segment {self._shm_name} is too small "
                f"({shm_stat.st_size} bytes). Expected at least {self.CTRL_SIZE}."
            )

        fd = os.open(self._shm_path, os.O_RDWR)
        try:
            self._shm = mmap.mmap(fd, self.CTRL_SIZE)
        finally:
            os.close(fd)

    def close(self) -> None:
        """Close the shared memory mapping."""
        if self._shm is not None:
            self._shm.close()
            self._shm = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> "GpuPoolController":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _read_control_block(self) -> tuple:
        """Read and unpack the control block."""
        self._shm.seek(0)
        data = self._shm.read(self.CTRL_SIZE)
        if len(data) != self.CTRL_SIZE:
            raise RuntimeError(
                f"Control block read size mismatch: expected {self.CTRL_SIZE}, "
                f"got {len(data)}"
            )
        return struct.unpack(self.CTRL_FORMAT, data)

    def _write_command(
        self,
        command: ResizeCommand,
        target_size: int = 0,
        delta: int = 0,
    ) -> None:
        """Write a command to the control block."""
        # Read current version
        current = self._read_control_block()
        new_version = current[self._IDX_VERSION] + 1

        # Pack and write the command fields
        # We only update: version, command, targetSizeBytes, deltaBytes
        self._shm.seek(0)

        # Write version
        self._shm.write(struct.pack("=I", new_version))
        # Write command
        self._shm.write(struct.pack("=I", command))
        # Skip status and reserved (4 bytes each), deviceId and reserved2 (4 bytes each)
        self._shm.seek(6 * 4)  # Skip to targetSizeBytes
        # Write targetSizeBytes
        self._shm.write(struct.pack("=q", target_size))
        # Write deltaBytes
        self._shm.write(struct.pack("=q", delta))

    def _wait_for_response(self, timeout_ms: int = 5000) -> dict:
        """
        Wait for the pool to process the command.

        Args:
            timeout_ms: Maximum time to wait in milliseconds.

        Returns:
            dict with keys: status, actual_size, available, error

        Raises:
            TimeoutError: If the command is not processed within the timeout.
        """
        start = time.time()
        while (time.time() - start) * 1000 < timeout_ms:
            data = self._read_control_block()
            # Command cleared means it has been processed
            if data[self._IDX_COMMAND] == ResizeCommand.NOP:
                error_bytes = data[self._IDX_ERROR_MSG]
                error_str = error_bytes.rstrip(b"\x00").decode("utf-8", errors="replace")
                return {
                    "status": ResizeStatus(data[self._IDX_STATUS]),
                    "actual_size": data[self._IDX_ACTUAL_SIZE],
                    "available": data[self._IDX_AVAILABLE],
                    "error": error_str,
                }
            time.sleep(0.001)  # 1ms poll interval

        raise TimeoutError(
            f"Resize command timed out after {timeout_ms}ms. "
            f"The pool may be unresponsive or IPC polling is disabled."
        )

    def query(self, timeout_ms: int = 1000) -> dict:
        """
        Query the current state of the memory pool.

        Args:
            timeout_ms: Maximum time to wait for response in milliseconds.

        Returns:
            dict with keys:
                - status: ResizeStatus enum value
                - actual_size: Current total pool size in bytes
                - available: Available (free) memory in bytes
                - error: Error message (empty string on success)

        Raises:
            TimeoutError: If the query is not processed within the timeout.
        """
        self._write_command(ResizeCommand.QUERY)
        return self._wait_for_response(timeout_ms)

    def expand(self, target_bytes: int, timeout_ms: int = 5000) -> dict:
        """
        Expand the memory pool to at least the specified size.

        The pool will allocate a new chunk to reach the target size.
        The actual size may be larger due to chunk alignment.

        Args:
            target_bytes: Target minimum pool size in bytes.
            timeout_ms: Maximum time to wait for response in milliseconds.

        Returns:
            dict with keys:
                - status: ResizeStatus.SUCCESS or ResizeStatus.FAILED
                - actual_size: Actual pool size after expansion
                - available: Available memory after expansion
                - error: Error message if failed

        Raises:
            TimeoutError: If the expansion is not processed within the timeout.
            ValueError: If target_bytes is negative.
        """
        if target_bytes < 0:
            raise ValueError("target_bytes must be non-negative")

        self._write_command(ResizeCommand.EXPAND_TO, target_size=target_bytes)
        return self._wait_for_response(timeout_ms)

    def expand_by(self, delta_bytes: int, timeout_ms: int = 5000) -> dict:
        """
        Expand the memory pool by the specified amount.

        Args:
            delta_bytes: Amount to expand the pool by in bytes.
            timeout_ms: Maximum time to wait for response in milliseconds.

        Returns:
            dict with keys:
                - status: ResizeStatus.SUCCESS or ResizeStatus.FAILED
                - actual_size: Actual pool size after expansion
                - available: Available memory after expansion
                - error: Error message if failed

        Raises:
            TimeoutError: If the expansion is not processed within the timeout.
            ValueError: If delta_bytes is negative.
        """
        if delta_bytes < 0:
            raise ValueError("delta_bytes must be non-negative")

        self._write_command(ResizeCommand.EXPAND_BY, delta=delta_bytes)
        return self._wait_for_response(timeout_ms)

    def shrink(self, target_bytes: int, timeout_ms: int = 5000) -> dict:
        """
        Shrink the memory pool to at most the specified size.

        The pool will only release chunks that are completely free (no active
        allocations). If some chunks cannot be released, the status will be
        ResizeStatus.PARTIAL.

        The pool cannot shrink below the currently used memory.

        Args:
            target_bytes: Target maximum pool size in bytes.
            timeout_ms: Maximum time to wait for response in milliseconds.

        Returns:
            dict with keys:
                - status: ResizeStatus.SUCCESS, PARTIAL, or FAILED
                - actual_size: Actual pool size after shrinking
                - available: Available memory after shrinking
                - error: Error message if failed/partial

        Raises:
            TimeoutError: If the shrink is not processed within the timeout.
            ValueError: If target_bytes is negative.
        """
        if target_bytes < 0:
            raise ValueError("target_bytes must be non-negative")

        self._write_command(ResizeCommand.SHRINK_TO, target_size=target_bytes)
        return self._wait_for_response(timeout_ms)

    def get_stats(self) -> dict:
        """
        Get detailed statistics about the memory pool.

        This is a convenience wrapper around query() that returns more
        descriptive statistics.

        Returns:
            dict with keys:
                - total_bytes: Total pool size
                - available_bytes: Available (free) memory
                - used_bytes: Currently allocated memory
                - utilization: Usage percentage (0.0 to 1.0)
        """
        result = self.query()
        total = result["actual_size"]
        available = result["available"]
        used = total - available

        return {
            "total_bytes": total,
            "available_bytes": available,
            "used_bytes": used,
            "utilization": used / total if total > 0 else 0.0,
        }


def get_pool_controller(device_id: int = 0) -> Optional[GpuPoolController]:
    """
    Get a pool controller for the specified device, or None if not available.

    This is a convenience function that catches FileNotFoundError and returns
    None instead, making it easier to check if IPC is available.

    Args:
        device_id: The GPU device ID.

    Returns:
        GpuPoolController instance or None if the pool doesn't exist.
    """
    try:
        return GpuPoolController(device_id)
    except FileNotFoundError:
        return None
