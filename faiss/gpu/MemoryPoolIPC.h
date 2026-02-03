// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace faiss {
namespace gpu {

/// Commands that can be sent via shared memory to control the preallocated
/// memory pool.
enum class ResizeCommand : uint32_t {
    Nop = 0,       /// No operation
    ExpandTo = 1,  /// Expand pool to at least `targetSizeBytes`
    ShrinkTo = 2,  /// Shrink pool down to (but not below usage) `targetSizeBytes`
    Query = 3,     /// Query current pool state
    ExpandBy = 4   /// Expand pool by `deltaBytes`
};

/// Status of a resize operation.
enum class ResizeStatus : uint32_t {
    Pending = 0,  /// Operation not yet processed
    Success = 1,  /// Operation completed successfully
    Failed = 2,   /// Operation failed
    Partial = 3   /// Partial success (e.g., shrink couldn't fully reach target)
};

/// Shared-memory control block layout for IPC communication.
///
/// This structure is intentionally POD and has a fixed layout that is mirrored
/// on the client side (e.g., Python) using `struct` with the following format:
///
///     CTRL_FORMAT = "=IIIIiiqqqq256s"
///
/// Memory layout (total 312 bytes):
///   Offset  Size  Field
///   ------  ----  -----
///      0      4   version (uint32_t)
///      4      4   command (uint32_t, ResizeCommand)
///      8      4   status (uint32_t, ResizeStatus)
///     12      4   reserved (uint32_t, padding)
///     16      4   deviceId (int32_t)
///     20      4   reserved2 (int32_t, padding)
///     24      8   targetSizeBytes (int64_t)
///     32      8   deltaBytes (int64_t)
///     40      8   actualSizeBytes (int64_t)
///     48      8   availableBytes (int64_t)
///     56    256   errorMsg (char[256])
///
/// Protocol:
///   1. Client increments `version` and sets `command` + parameters
///   2. Server detects version change, processes command
///   3. Server updates `status`, `actualSizeBytes`, `availableBytes`, `errorMsg`
///   4. Server clears `command` to Nop to signal completion
///   5. Client polls until `command == Nop`, then reads response
///
/// Shared memory path convention: /faiss_gpu_pool_ctrl_{device_id}
///   e.g., GPU 0 uses /faiss_gpu_pool_ctrl_0
struct ShmControlBlock {
    /// Version number, incremented by client for each new command
    uint32_t version;

    /// Command to execute (ResizeCommand enum)
    uint32_t command;

    /// Status of last operation (ResizeStatus enum)
    uint32_t status;

    /// Reserved for alignment (unused)
    uint32_t reserved;

    /// Target GPU device ID
    int32_t deviceId;

    /// Reserved for alignment (unused)
    int32_t reserved2;

    /// Target size in bytes for ExpandTo/ShrinkTo commands
    int64_t targetSizeBytes;

    /// Delta size in bytes for ExpandBy command
    int64_t deltaBytes;

    /// Actual pool size after operation (response field)
    int64_t actualSizeBytes;

    /// Available (free) bytes in pool (response field)
    int64_t availableBytes;

    /// Error message if operation failed (null-terminated)
    char errorMsg[256];

    /// Initialize control block to default state
    void init(int device) {
        std::memset(this, 0, sizeof(ShmControlBlock));
        deviceId = device;
    }

    /// Set error message safely (ensures null-termination)
    void setErrorMsg(const char* msg) {
        if (msg) {
            std::strncpy(errorMsg, msg, sizeof(errorMsg) - 1);
            errorMsg[sizeof(errorMsg) - 1] = '\0';
        } else {
            errorMsg[0] = '\0';
        }
    }

    /// Set error message from std::string
    void setErrorMsg(const std::string& msg) {
        setErrorMsg(msg.c_str());
    }

    /// Clear error message
    void clearErrorMsg() {
        errorMsg[0] = '\0';
    }
};

// Verify struct size matches expected layout (312 bytes)
static_assert(sizeof(ShmControlBlock) == 312,
              "ShmControlBlock size mismatch - check alignment");

// Verify field offsets match expected layout
static_assert(offsetof(ShmControlBlock, version) == 0, "version offset mismatch");
static_assert(offsetof(ShmControlBlock, command) == 4, "command offset mismatch");
static_assert(offsetof(ShmControlBlock, status) == 8, "status offset mismatch");
static_assert(offsetof(ShmControlBlock, reserved) == 12, "reserved offset mismatch");
static_assert(offsetof(ShmControlBlock, deviceId) == 16, "deviceId offset mismatch");
static_assert(offsetof(ShmControlBlock, reserved2) == 20, "reserved2 offset mismatch");
static_assert(offsetof(ShmControlBlock, targetSizeBytes) == 24, "targetSizeBytes offset mismatch");
static_assert(offsetof(ShmControlBlock, deltaBytes) == 32, "deltaBytes offset mismatch");
static_assert(offsetof(ShmControlBlock, actualSizeBytes) == 40, "actualSizeBytes offset mismatch");
static_assert(offsetof(ShmControlBlock, availableBytes) == 48, "availableBytes offset mismatch");
static_assert(offsetof(ShmControlBlock, errorMsg) == 56, "errorMsg offset mismatch");

/// Result of a resize/query operation on the pool.
/// Used internally on the server side (not shared via IPC).
struct ResizeResult {
    ResizeStatus status;
    size_t actualSize;
    size_t availableSize;
    std::string errorMsg;

    ResizeResult()
            : status(ResizeStatus::Pending),
              actualSize(0),
              availableSize(0),
              errorMsg() {}

    ResizeResult(
            ResizeStatus s,
            size_t actual,
            size_t available,
            const std::string& err = "")
            : status(s),
              actualSize(actual),
              availableSize(available),
              errorMsg(err) {}

    /// Check if operation was successful
    bool isSuccess() const {
        return status == ResizeStatus::Success;
    }

    /// Check if operation partially succeeded (for shrink operations)
    bool isPartial() const {
        return status == ResizeStatus::Partial;
    }

    /// Check if operation failed
    bool isFailed() const {
        return status == ResizeStatus::Failed;
    }
};

/// Shared memory name prefix for GPU pool control
constexpr const char* kShmNamePrefix = "/faiss_gpu_pool_ctrl_";

/// Generate shared memory name for a specific device
inline std::string getShmName(int deviceId) {
    return std::string(kShmNamePrefix) + std::to_string(deviceId);
}

} // namespace gpu
} // namespace faiss
