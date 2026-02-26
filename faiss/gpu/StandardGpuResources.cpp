// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#if defined USE_NVIDIA_CUVS
#include <raft/core/device_resources.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/host/pinned_memory_resource.hpp>
#include <memory>
#endif

#include <faiss/gpu/MemoryPoolIPC.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/impl/FaissAssert.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <queue>
#include <sstream>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace faiss {
namespace gpu {

namespace {

// How many streams per device we allocate by default (for multi-streaming)
constexpr int kNumStreams = 2;

// Use 256 MiB of pinned memory for async CPU <-> GPU copies by default
constexpr size_t kDefaultPinnedMemoryAllocation = (size_t)256 * 1024 * 1024;

// Default temporary memory allocation for <= 4 GiB memory GPUs
constexpr size_t k4GiBTempMem = (size_t)512 * 1024 * 1024;

// Default temporary memory allocation for <= 8 GiB memory GPUs
constexpr size_t k8GiBTempMem = (size_t)1024 * 1024 * 1024;

// Maximum temporary memory allocation for all GPUs
constexpr size_t kMaxTempMem = (size_t)1536 * 1024 * 1024;

std::string allocsToString(const std::unordered_map<void*, AllocRequest>& map) {
    // Produce a sorted list of all outstanding allocations by type
    std::unordered_map<AllocType, std::pair<int, size_t>> stats;

    for (auto& entry : map) {
        auto& a = entry.second;

        auto it = stats.find(a.type);
        if (it != stats.end()) {
            stats[a.type].first++;
            stats[a.type].second += a.size;
        } else {
            stats[a.type] = std::make_pair(1, a.size);
        }
    }

    std::stringstream ss;
    for (auto& entry : stats) {
        ss << "Alloc type " << allocTypeToString(entry.first) << ": "
           << entry.second.first << " allocations, " << entry.second.second
           << " bytes\n";
    }

    return ss.str();
}

} // namespace

// Default minimum chunk size for elastic scaling (256 MiB)
constexpr size_t kDefaultMinChunkSize = (size_t)256 * 1024 * 1024;

// IPC polling interval in milliseconds
constexpr int kIPCPollIntervalMs = 10;

// Async expansion check interval in milliseconds
constexpr int kAsyncExpandCheckIntervalMs = 5;

/// Request for async expansion
struct AsyncExpandRequest {
    size_t targetSize;
    uint64_t requestId;
    std::chrono::steady_clock::time_point timestamp;
};

/// Status of async expansion
enum class AsyncExpandStatus {
    Idle,        // No expansion in progress
    Pending,     // Expansion requested, waiting to start
    InProgress,  // Expansion is currently being processed
    Completed,   // Expansion completed successfully
    Failed       // Expansion failed
};

/// A single memory chunk within the preallocated pool.
struct MemoryChunk {
    void* base;      // Base address of this chunk
    size_t size;     // Total size of this chunk
    size_t usedBytes; // Currently allocated bytes within this chunk
    int chunkId;     // Unique identifier
    std::map<char*, size_t> freeBlocks; // Free blocks within this chunk

    MemoryChunk() : base(nullptr), size(0), usedBytes(0), chunkId(-1) {}

    bool isFullyFree() const {
        return usedBytes == 0;
    }

    bool owns(const void* p) const {
        if (!base || !p) {
            return false;
        }
        const char* basePtr = static_cast<const char*>(base);
        const char* ptr = static_cast<const char*>(p);
        return ptr >= basePtr && ptr < (basePtr + size);
    }
};

// NOTE:(wangzehao) PreallocMemoryPool is a class that manages a pool of memory
// with support for online elastic scaling via shared memory IPC.
// Supports async expansion that runs in background without blocking user requests.
class PreallocMemoryPool {
   public:
    PreallocMemoryPool(int device, size_t initialSize, bool enableIPC = true)
            : device_(device),
              totalSize_(0),
              minChunkSize_(kDefaultMinChunkSize),
              nextChunkId_(0),
              shmFd_(-1),
              shmPtr_(nullptr),
              stopPolling_(false),
              nextAsyncRequestId_(0),
              asyncExpandStatus_(AsyncExpandStatus::Idle),
              stopAsyncExpand_(false) {
        // Allocate initial chunk if size > 0
        if (initialSize > 0) {
            size_t alignedSize = utils::roundUp(initialSize, (size_t)256);
            DeviceScope scope(device_);

            void* base = nullptr;
            auto err = cudaMalloc(&base, alignedSize);
            FAISS_ASSERT_FMT(
                    err == cudaSuccess,
                    "Failed to pre-allocate device memory pool (size %zu) on "
                    "device %d (error %d %s)",
                    alignedSize,
                    device_,
                    (int)err,
                    cudaGetErrorString(err));

            auto chunk = std::make_unique<MemoryChunk>();
            chunk->base = base;
            chunk->size = alignedSize;
            chunk->usedBytes = 0;
            chunk->chunkId = nextChunkId_++;
            chunk->freeBlocks.emplace(static_cast<char*>(base), alignedSize);

            chunks_.push_back(std::move(chunk));
            totalSize_ = alignedSize;
        }

        // Initialize shared memory IPC if enabled
        if (enableIPC) {
            initSharedMemory();
            startPollingThread();
        }

        // Start async expansion thread
        startAsyncExpandThread();
    }

    ~PreallocMemoryPool() {
        // Stop async expansion thread first
        stopAsyncExpandThread();

        // Stop the polling thread
        stopPollingThread();

        // Clean up shared memory
        cleanupSharedMemory();

        // Free all chunks
        DeviceScope scope(device_);
        for (auto& chunk : chunks_) {
            if (chunk && chunk->base) {
                auto err = cudaFree(chunk->base);
                if (err != cudaSuccess) {
                    std::cerr << "Warning: Failed to free chunk " << chunk->chunkId
                              << " on device " << device_ << " (error "
                              << cudaGetErrorString(err) << ")\n";
                }
            }
        }
        chunks_.clear();
        totalSize_ = 0;
    }

    /// Check if pointer belongs to any chunk in the pool
    bool owns(const void* p) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return ownsLocked(p);
    }

    /// Get total available (free) memory across all chunks
    size_t getSizeAvailable() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return getSizeAvailableLocked();
    }

    /// Get total pool size
    size_t getTotalSize() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return totalSize_;
    }

    /// Allocate memory from the pool (first-fit across chunks)
    void* allocate(size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        return allocateLocked(size);
    }

    /// Deallocate memory back to the pool
    void deallocate(void* p, size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        deallocateLocked(p, size);
    }

    /// Expand the pool to at least targetSize bytes
    ResizeResult expand(size_t targetSize) {
        std::lock_guard<std::mutex> lock(mutex_);
        return expandLocked(targetSize);
    }

    /// Shrink the pool to at most targetSize bytes (only frees fully-free chunks)
    ResizeResult shrink(size_t targetSize) {
        std::lock_guard<std::mutex> lock(mutex_);
        return shrinkLocked(targetSize);
    }

    /// Query current pool state
    ResizeResult query() {
        std::lock_guard<std::mutex> lock(mutex_);
        return {ResizeStatus::Success, totalSize_, getSizeAvailableLocked(), ""};
    }

    /// Submit an async expansion request (non-blocking)
    /// Returns a request ID that can be used to query the status
    uint64_t expandAsync(size_t targetSize) {
        std::lock_guard<std::mutex> lock(asyncMutex_);
        
        uint64_t requestId = nextAsyncRequestId_++;
        AsyncExpandRequest req;
        req.targetSize = targetSize;
        req.requestId = requestId;
        req.timestamp = std::chrono::steady_clock::now();
        
        asyncExpandQueue_.push(req);
        asyncExpandCv_.notify_one();
        
        return requestId;
    }

    /// Query the status of async expansion
    AsyncExpandStatus getAsyncExpandStatus() const {
        return asyncExpandStatus_.load(std::memory_order_acquire);
    }

    /// Get the last async expansion result (thread-safe)
    ResizeResult getLastAsyncExpandResult() const {
        std::lock_guard<std::mutex> lock(asyncMutex_);
        return lastAsyncExpandResult_;
    }

    /// Check if there's a pending async expansion
    bool hasAsyncExpandPending() const {
        std::lock_guard<std::mutex> lock(asyncMutex_);
        return !asyncExpandQueue_.empty() || 
               asyncExpandStatus_.load(std::memory_order_acquire) == AsyncExpandStatus::InProgress;
    }

   private:
    bool ownsLocked(const void* p) const {
        for (const auto& chunk : chunks_) {
            if (chunk && chunk->owns(p)) {
                return true;
            }
        }
        return false;
    }

    size_t getSizeAvailableLocked() const {
        size_t total = 0;
        for (const auto& chunk : chunks_) {
            for (const auto& entry : chunk->freeBlocks) {
                total += entry.second;
            }
        }
        return total;
    }

    void* allocateLocked(size_t size) {
        if (size == 0 || totalSize_ == 0) {
            return nullptr;
        }

        size = utils::roundUp(size, (size_t)256);

        // First-fit across all chunks
        for (auto& chunk : chunks_) {
            for (auto it = chunk->freeBlocks.begin();
                 it != chunk->freeBlocks.end();
                 ++it) {
                if (it->second >= size) {
                    char* ptr = it->first;
                    size_t remaining = it->second - size;
                    chunk->freeBlocks.erase(it);
                    if (remaining > 0) {
                        chunk->freeBlocks.emplace(ptr + size, remaining);
                    }
                    chunk->usedBytes += size;
                    return ptr;
                }
            }
        }
        return nullptr; // OOM within pool
    }

    void deallocateLocked(void* p, size_t size) {
        if (!p || totalSize_ == 0) {
            return;
        }

        char* ptr = static_cast<char*>(p);
        size = utils::roundUp(size, (size_t)256);

        // Find the owning chunk
        MemoryChunk* ownerChunk = nullptr;
        for (auto& chunk : chunks_) {
            if (chunk->owns(ptr)) {
                ownerChunk = chunk.get();
                break;
            }
        }

        FAISS_ASSERT_FMT(
                ownerChunk != nullptr,
                "Pointer %p does not belong to any chunk on device %d",
                p,
                device_);

        char* chunkBase = static_cast<char*>(ownerChunk->base);
        FAISS_ASSERT_FMT(
                ptr + size <= chunkBase + ownerChunk->size,
                "Invalid deallocation: pointer %p with size %zu exceeds chunk bounds",
                p,
                size);

        auto& freeBlocks = ownerChunk->freeBlocks;

        // Merge with adjacent free blocks
        auto next = freeBlocks.lower_bound(ptr);
        if (next != freeBlocks.begin()) {
            auto prev = std::prev(next);
            FAISS_ASSERT(prev->first + prev->second <= ptr);
            if (prev->first + prev->second == ptr) {
                ptr = prev->first;
                size += prev->second;
                freeBlocks.erase(prev);
            }
        }

        if (next != freeBlocks.end()) {
            FAISS_ASSERT(ptr + size <= next->first);
            if (ptr + size == next->first) {
                size += next->second;
                freeBlocks.erase(next);
            }
        }

        freeBlocks.emplace(ptr, size);
        ownerChunk->usedBytes -= utils::roundUp(
                static_cast<size_t>(static_cast<char*>(p) - ptr == 0 ? size : size),
                (size_t)256);
        
        // Recalculate usedBytes properly
        size_t totalFree = 0;
        for (const auto& fb : ownerChunk->freeBlocks) {
            totalFree += fb.second;
        }
        ownerChunk->usedBytes = ownerChunk->size - totalFree;
    }

    ResizeResult expandLocked(size_t targetSize) {
        if (targetSize <= totalSize_) {
            return {ResizeStatus::Success,
                    totalSize_,
                    getSizeAvailableLocked(),
                    ""};
        }

        size_t delta = targetSize - totalSize_;
        delta = utils::roundUp(delta, minChunkSize_);

        DeviceScope scope(device_);
        void* newBase = nullptr;
        auto err = cudaMalloc(&newBase, delta);

        if (err != cudaSuccess) {
            cudaGetLastError(); // Clear error
            return {ResizeStatus::Failed,
                    totalSize_,
                    getSizeAvailableLocked(),
                    std::string("cudaMalloc failed: ") + cudaGetErrorString(err)};
        }

        auto chunk = std::make_unique<MemoryChunk>();
        chunk->base = newBase;
        chunk->size = delta;
        chunk->usedBytes = 0;
        chunk->chunkId = nextChunkId_++;
        chunk->freeBlocks.emplace(static_cast<char*>(newBase), delta);

        chunks_.push_back(std::move(chunk));
        totalSize_ += delta;

        return {ResizeStatus::Success, totalSize_, getSizeAvailableLocked(), ""};
    }

    /// Perform expansion without holding the main mutex during cudaMalloc.
    /// This allows allocations to continue from existing chunks during expansion.
    ResizeResult expandNonBlocking(size_t targetSize) {
        // Phase 1: Check if expansion is needed and calculate delta (with lock)
        size_t delta = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (targetSize <= totalSize_) {
                return {ResizeStatus::Success,
                        totalSize_,
                        getSizeAvailableLocked(),
                        ""};
            }
            delta = targetSize - totalSize_;
            delta = utils::roundUp(delta, minChunkSize_);
        }

        // Phase 2: Allocate memory WITHOUT holding the lock
        // This is the slow operation that we don't want to block other threads
        DeviceScope scope(device_);
        void* newBase = nullptr;
        auto err = cudaMalloc(&newBase, delta);

        if (err != cudaSuccess) {
            cudaGetLastError(); // Clear error
            std::lock_guard<std::mutex> lock(mutex_);
            return {ResizeStatus::Failed,
                    totalSize_,
                    getSizeAvailableLocked(),
                    std::string("cudaMalloc failed: ") + cudaGetErrorString(err)};
        }

        // Phase 3: Add new chunk to pool (with lock)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            
            // Double-check: another thread may have expanded while we were allocating
            // In that case, we still add our chunk (more memory doesn't hurt)
            auto chunk = std::make_unique<MemoryChunk>();
            chunk->base = newBase;
            chunk->size = delta;
            chunk->usedBytes = 0;
            chunk->chunkId = nextChunkId_++;
            chunk->freeBlocks.emplace(static_cast<char*>(newBase), delta);

            chunks_.push_back(std::move(chunk));
            totalSize_ += delta;

            return {ResizeStatus::Success, totalSize_, getSizeAvailableLocked(), ""};
        }
    }

    ResizeResult shrinkLocked(size_t targetSize) {
        if (targetSize >= totalSize_) {
            return {ResizeStatus::Success,
                    totalSize_,
                    getSizeAvailableLocked(),
                    ""};
        }

        size_t currentUsed = totalSize_ - getSizeAvailableLocked();
        if (targetSize < currentUsed) {
            return {ResizeStatus::Failed,
                    totalSize_,
                    getSizeAvailableLocked(),
                    "Target size smaller than current usage"};
        }

        // Collect fully-free chunk indices
        std::vector<size_t> freeableIndices;
        for (size_t i = 0; i < chunks_.size(); ++i) {
            if (chunks_[i]->isFullyFree()) {
                freeableIndices.push_back(i);
            }
        }

        // Sort by size ascending (prefer releasing smaller chunks first)
        std::sort(
                freeableIndices.begin(),
                freeableIndices.end(),
                [this](size_t a, size_t b) {
                    return chunks_[a]->size < chunks_[b]->size;
                });

        DeviceScope scope(device_);

        // Synchronize to ensure no kernels are using the memory
        cudaDeviceSynchronize();

        size_t toRelease = totalSize_ - targetSize;
        size_t released = 0;
        std::vector<size_t> toRemove;

        for (size_t idx : freeableIndices) {
            if (released >= toRelease) {
                break;
            }

            auto& chunk = chunks_[idx];
            auto err = cudaFree(chunk->base);
            if (err != cudaSuccess) {
                // Log warning but continue
                std::cerr << "Warning: Failed to free chunk " << chunk->chunkId
                          << " (error " << cudaGetErrorString(err) << ")\n";
                continue;
            }
            released += chunk->size;
            toRemove.push_back(idx);
        }

        // Remove chunks from back to front to keep indices valid
        std::sort(toRemove.rbegin(), toRemove.rend());
        for (size_t idx : toRemove) {
            chunks_.erase(chunks_.begin() + idx);
        }
        totalSize_ -= released;

        if (released >= toRelease) {
            return {ResizeStatus::Success,
                    totalSize_,
                    getSizeAvailableLocked(),
                    ""};
        } else {
            return {ResizeStatus::Partial,
                    totalSize_,
                    getSizeAvailableLocked(),
                    "Some chunks still in use"};
        }
    }

    void initSharedMemory() {
        std::string shmName = getShmName(device_);

        shmFd_ = shm_open(shmName.c_str(), O_CREAT | O_RDWR, 0666);
        if (shmFd_ < 0) {
            std::cerr << "Warning: Failed to open shared memory " << shmName
                      << ": " << strerror(errno) << "\n";
            return;
        }

        if (ftruncate(shmFd_, sizeof(ShmControlBlock)) < 0) {
            std::cerr << "Warning: Failed to resize shared memory: "
                      << strerror(errno) << "\n";
            close(shmFd_);
            shmFd_ = -1;
            return;
        }

        shmPtr_ = mmap(
                nullptr,
                sizeof(ShmControlBlock),
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                shmFd_,
                0);
        if (shmPtr_ == MAP_FAILED) {
            std::cerr << "Warning: Failed to mmap shared memory: "
                      << strerror(errno) << "\n";
            close(shmFd_);
            shmFd_ = -1;
            shmPtr_ = nullptr;
            return;
        }

        // Initialize control block using helper method
        auto* ctrl = static_cast<ShmControlBlock*>(shmPtr_);
        ctrl->init(device_);
        ctrl->actualSizeBytes = static_cast<int64_t>(totalSize_);
        ctrl->availableBytes = static_cast<int64_t>(getSizeAvailableLocked());
    }

    void cleanupSharedMemory() {
        if (shmPtr_ && shmPtr_ != MAP_FAILED) {
            munmap(shmPtr_, sizeof(ShmControlBlock));
            shmPtr_ = nullptr;
        }
        if (shmFd_ >= 0) {
            close(shmFd_);
            // Optionally unlink (but leave it for potential reconnection)
            // std::string shmName = "/faiss_gpu_pool_ctrl_" + std::to_string(device_);
            // shm_unlink(shmName.c_str());
            shmFd_ = -1;
        }
    }

    void startPollingThread() {
        if (!shmPtr_) {
            return; // IPC not initialized
        }

        stopPolling_.store(false, std::memory_order_release);
        pollingThread_ = std::thread([this]() { pollLoop(); });
    }

    void stopPollingThread() {
        stopPolling_.store(true, std::memory_order_release);
        if (pollingThread_.joinable()) {
            pollingThread_.join();
        }
    }

    void pollLoop() {
        if (!shmPtr_) {
            return;
        }

        auto* ctrl = static_cast<ShmControlBlock*>(shmPtr_);
        uint32_t lastVersion = 0;

        while (!stopPolling_.load(std::memory_order_acquire)) {
            uint32_t curVersion = ctrl->version;
            // Use a memory fence to ensure we see consistent data
            std::atomic_thread_fence(std::memory_order_acquire);

            if (curVersion != lastVersion && ctrl->command != 0) {
                lastVersion = curVersion;
                processCommand(ctrl);
            }

            std::this_thread::sleep_for(
                    std::chrono::milliseconds(kIPCPollIntervalMs));
        }
    }

    void processCommand(ShmControlBlock* ctrl) {
        auto cmd = static_cast<ResizeCommand>(ctrl->command);
        ResizeResult result;

        switch (cmd) {
            case ResizeCommand::ExpandTo: {
                // Use async expansion for expand commands
                size_t targetSize = static_cast<size_t>(ctrl->targetSizeBytes);
                size_t currentSize = getTotalSize();
                
                if (targetSize > currentSize) {
                    // Submit async expand request
                    expandAsync(targetSize);
                    
                    // Return immediately with pending status
                    result.status = ResizeStatus::Pending;
                    result.actualSize = currentSize;
                    result.availableSize = getSizeAvailable();
                    result.errorMsg = "Async expansion submitted";
                } else {
                    // No expansion needed
                    result = query();
                }
                break;
            }
            case ResizeCommand::ShrinkTo:
                result = shrink(static_cast<size_t>(ctrl->targetSizeBytes));
                break;
            case ResizeCommand::Query: {
                result = query();
                // Also include async expand status in the response
                auto asyncStatus = getAsyncExpandStatus();
                if (asyncStatus == AsyncExpandStatus::InProgress ||
                    asyncStatus == AsyncExpandStatus::Pending) {
                    result.errorMsg = "Async expansion in progress";
                }
                break;
            }
            case ResizeCommand::ExpandBy: {
                // Use async expansion for expand-by commands
                size_t currentSize = getTotalSize();
                size_t targetSize = currentSize + static_cast<size_t>(ctrl->deltaBytes);
                
                expandAsync(targetSize);
                
                result.status = ResizeStatus::Pending;
                result.actualSize = currentSize;
                result.availableSize = getSizeAvailable();
                result.errorMsg = "Async expansion submitted";
                break;
            }
            case ResizeCommand::Nop:
            default:
                return;
        }

        // Write response
        ctrl->status = static_cast<uint32_t>(result.status);
        ctrl->actualSizeBytes = static_cast<int64_t>(result.actualSize);
        ctrl->availableBytes = static_cast<int64_t>(result.availableSize);
        ctrl->setErrorMsg(result.errorMsg);

        // Memory fence before clearing command
        std::atomic_thread_fence(std::memory_order_release);

        // Clear command to signal completion
        ctrl->command = static_cast<uint32_t>(ResizeCommand::Nop);
    }

    void startAsyncExpandThread() {
        stopAsyncExpand_.store(false, std::memory_order_release);
        asyncExpandThread_ = std::thread([this]() { asyncExpandLoop(); });
    }

    void stopAsyncExpandThread() {
        stopAsyncExpand_.store(true, std::memory_order_release);
        asyncExpandCv_.notify_all();
        if (asyncExpandThread_.joinable()) {
            asyncExpandThread_.join();
        }
    }

    void asyncExpandLoop() {
        while (!stopAsyncExpand_.load(std::memory_order_acquire)) {
            AsyncExpandRequest req;
            bool hasRequest = false;
            
            {
                std::unique_lock<std::mutex> lock(asyncMutex_);
                
                // Wait for a request or stop signal
                asyncExpandCv_.wait_for(
                    lock,
                    std::chrono::milliseconds(kAsyncExpandCheckIntervalMs),
                    [this]() {
                        return !asyncExpandQueue_.empty() || 
                               stopAsyncExpand_.load(std::memory_order_acquire);
                    });
                
                if (stopAsyncExpand_.load(std::memory_order_acquire)) {
                    break;
                }
                
                if (!asyncExpandQueue_.empty()) {
                    req = asyncExpandQueue_.front();
                    asyncExpandQueue_.pop();
                    hasRequest = true;
                    asyncExpandStatus_.store(AsyncExpandStatus::InProgress, std::memory_order_release);
                }
            }
            
            if (hasRequest) {
                // Perform the expansion using non-blocking method
                // This doesn't hold mutex_ during cudaMalloc, allowing other
                // allocations to continue using existing pool memory
                ResizeResult result = expandNonBlocking(req.targetSize);
                
                // Update status and result
                {
                    std::lock_guard<std::mutex> lock(asyncMutex_);
                    lastAsyncExpandResult_ = result;
                    
                    if (result.isSuccess()) {
                        asyncExpandStatus_.store(AsyncExpandStatus::Completed, std::memory_order_release);
                    } else {
                        asyncExpandStatus_.store(AsyncExpandStatus::Failed, std::memory_order_release);
                    }
                }
                
                // Update shared memory status if available
                if (shmPtr_) {
                    auto* ctrl = static_cast<ShmControlBlock*>(shmPtr_);
                    ctrl->actualSizeBytes = static_cast<int64_t>(result.actualSize);
                    ctrl->availableBytes = static_cast<int64_t>(result.availableSize);
                    ctrl->status = static_cast<uint32_t>(result.status);
                    std::atomic_thread_fence(std::memory_order_release);
                }
                
                // Reset to idle after a short delay if no more requests
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                {
                    std::lock_guard<std::mutex> lock(asyncMutex_);
                    if (asyncExpandQueue_.empty()) {
                        asyncExpandStatus_.store(AsyncExpandStatus::Idle, std::memory_order_release);
                    }
                }
            }
        }
    }

   private:
    int device_;
    mutable std::mutex mutex_;
    std::vector<std::unique_ptr<MemoryChunk>> chunks_;
    size_t totalSize_;
    size_t minChunkSize_;
    int nextChunkId_;

    // IPC shared memory
    int shmFd_;
    void* shmPtr_;
    std::atomic<bool> stopPolling_;
    std::thread pollingThread_;

    // Async expansion support
    mutable std::mutex asyncMutex_;
    std::condition_variable asyncExpandCv_;
    std::queue<AsyncExpandRequest> asyncExpandQueue_;
    std::thread asyncExpandThread_;
    std::atomic<bool> stopAsyncExpand_;
    uint64_t nextAsyncRequestId_;
    std::atomic<AsyncExpandStatus> asyncExpandStatus_;
    ResizeResult lastAsyncExpandResult_;
};

//
// StandardGpuResourcesImpl
//

StandardGpuResourcesImpl::StandardGpuResourcesImpl()
        :
#if defined USE_NVIDIA_CUVS
          mmr_(new rmm::mr::managed_memory_resource),
          pmr_(new rmm::mr::pinned_memory_resource),
#endif
          pinnedMemAlloc_(nullptr),
          pinnedMemAllocSize_(0),
          // let the adjustment function determine the memory size for us by
          // passing in a huge value that will then be adjusted
          tempMemSize_(getDefaultTempMemForGPU(
                  -1,
                  std::numeric_limits<size_t>::max())),
          deviceMemSize_(0),
          pinnedMemSize_(kDefaultPinnedMemoryAllocation),
          allocLogging_(false) {
}

StandardGpuResourcesImpl::~StandardGpuResourcesImpl() {
    // The temporary memory allocator has allocated memory through us, so clean
    // that up before we finish fully de-initializing ourselves
    tempMemory_.clear();

    // Make sure all allocations have been freed
    bool allocError = false;

    for (auto& entry : allocs_) {
        auto& map = entry.second;

        if (!map.empty()) {
            std::cerr
                    << "StandardGpuResources destroyed with allocations outstanding:\n"
                    << "Device " << entry.first
                    << " outstanding allocations:\n";
            std::cerr << allocsToString(map);
            allocError = true;
        }
    }

    FAISS_ASSERT_MSG(
            !allocError, "GPU memory allocations not properly cleaned up");

    preallocPools_.clear();

#if defined USE_NVIDIA_CUVS
    raftHandles_.clear();
#endif

    for (auto& entry : defaultStreams_) {
        DeviceScope scope(entry.first);

        // We created these streams, so are responsible for destroying them
        CUDA_VERIFY(cudaStreamDestroy(entry.second));
    }

    for (auto& entry : alternateStreams_) {
        DeviceScope scope(entry.first);

        for (auto stream : entry.second) {
            CUDA_VERIFY(cudaStreamDestroy(stream));
        }
    }

    for (auto& entry : asyncCopyStreams_) {
        DeviceScope scope(entry.first);

        CUDA_VERIFY(cudaStreamDestroy(entry.second));
    }

    for (auto& entry : blasHandles_) {
        DeviceScope scope(entry.first);

        auto blasStatus = cublasDestroy(entry.second);
        FAISS_ASSERT(blasStatus == CUBLAS_STATUS_SUCCESS);
    }

    if (pinnedMemAlloc_) {
#if defined USE_NVIDIA_CUVS
        pmr_->deallocate(pinnedMemAlloc_, pinnedMemAllocSize_);
#else
        auto err = cudaFreeHost(pinnedMemAlloc_);
        FAISS_ASSERT_FMT(
                err == cudaSuccess,
                "Failed to cudaFreeHost pointer %p (error %d %s)",
                pinnedMemAlloc_,
                (int)err,
                cudaGetErrorString(err));
#endif
    }
}

size_t StandardGpuResourcesImpl::getDefaultTempMemForGPU(
        int device,
        size_t requested) {
    auto totalMem = device != -1 ? getDeviceProperties(device).totalGlobalMem
                                 : std::numeric_limits<size_t>::max();

    if (totalMem <= (size_t)4 * 1024 * 1024 * 1024) {
        // If the GPU has <= 4 GiB of memory, reserve 512 MiB

        if (requested > k4GiBTempMem) {
            return k4GiBTempMem;
        }
    } else if (totalMem <= (size_t)8 * 1024 * 1024 * 1024) {
        // If the GPU has <= 8 GiB of memory, reserve 1 GiB

        if (requested > k8GiBTempMem) {
            return k8GiBTempMem;
        }
    } else {
        // Never use more than 1.5 GiB
        if (requested > kMaxTempMem) {
            return kMaxTempMem;
        }
    }

    // use whatever lower limit the user requested
    return requested;
}

/// Does the given GPU support bfloat16?
bool StandardGpuResourcesImpl::supportsBFloat16(int device) {
    initializeForDevice(device);
    auto& prop = getDeviceProperties(device);
    return prop.major >= 8;
}

void StandardGpuResourcesImpl::noTempMemory() {
    setTempMemory(0);
}

void StandardGpuResourcesImpl::setTempMemory(size_t size) {
    if (tempMemSize_ != size) {
        // adjust based on general limits
        tempMemSize_ = getDefaultTempMemForGPU(-1, size);

        // We need to re-initialize memory resources for all current devices
        // that have been initialized. This should be safe to do, even if we are
        // currently running work, because the cudaFree call that this implies
        // will force-synchronize all GPUs with the CPU
        for (auto& p : tempMemory_) {
            int device = p.first;
            // Free the existing memory first
            p.second.reset();

            // Allocate new
            p.second = std::make_unique<StackDeviceMemory>(
                    this,
                    p.first,
                    // adjust for this specific device
                    getDefaultTempMemForGPU(device, tempMemSize_));
        }
    }
}

void StandardGpuResourcesImpl::setDeviceMemoryReservation(size_t size) {
    // Should not call this after devices have been initialized
    FAISS_ASSERT(defaultStreams_.size() == 0);
    FAISS_ASSERT(preallocPools_.empty());

    deviceMemSize_ = size;
}

void StandardGpuResourcesImpl::setPinnedMemory(size_t size) {
    // Should not call this after devices have been initialized
    FAISS_ASSERT(defaultStreams_.size() == 0);
    FAISS_ASSERT(!pinnedMemAlloc_);

    pinnedMemSize_ = size;
}

void StandardGpuResourcesImpl::setDefaultStream(
        int device,
        cudaStream_t stream) {
    if (isInitialized(device)) {
        // A new series of calls may not be ordered with what was the previous
        // stream, so if the stream being specified is different, then we need
        // to ensure ordering between the two (new stream waits on old).
        auto it = userDefaultStreams_.find(device);
        cudaStream_t prevStream = nullptr;

        if (it != userDefaultStreams_.end()) {
            prevStream = it->second;
        } else {
            FAISS_ASSERT(defaultStreams_.count(device));
            prevStream = defaultStreams_[device];
        }

        if (prevStream != stream) {
            streamWait({stream}, {prevStream});
        }
#if defined USE_NVIDIA_CUVS
        // delete the raft handle for this device, which will be initialized
        // with the updated stream during any subsequent calls to getRaftHandle
        auto it2 = raftHandles_.find(device);
        if (it2 != raftHandles_.end()) {
            raft::resource::set_cuda_stream(it2->second, stream);
        }
#endif
    }

    userDefaultStreams_[device] = stream;
}

void StandardGpuResourcesImpl::revertDefaultStream(int device) {
    if (isInitialized(device)) {
        auto it = userDefaultStreams_.find(device);

        if (it != userDefaultStreams_.end()) {
            // There was a user stream set that we need to synchronize against
            cudaStream_t prevStream = userDefaultStreams_[device];

            FAISS_ASSERT(defaultStreams_.count(device));
            cudaStream_t newStream = defaultStreams_[device];

            streamWait({newStream}, {prevStream});

#if defined USE_NVIDIA_CUVS
            // update the stream on the raft handle for this device
            auto it2 = raftHandles_.find(device);
            if (it2 != raftHandles_.end()) {
                raft::resource::set_cuda_stream(it2->second, newStream);
            }
#endif
        } else {
#if defined USE_NVIDIA_CUVS
            // delete the raft handle for this device, which will be initialized
            // with the updated stream during any subsequent calls to
            // getRaftHandle
            auto it2 = raftHandles_.find(device);
            if (it2 != raftHandles_.end()) {
                raftHandles_.erase(it2);
            }
#endif
        }
    }

    userDefaultStreams_.erase(device);
}

void StandardGpuResourcesImpl::setDefaultNullStreamAllDevices() {
    for (int dev = 0; dev < getNumDevices(); ++dev) {
        setDefaultStream(dev, nullptr);
    }
}

void StandardGpuResourcesImpl::setLogMemoryAllocations(bool enable) {
    allocLogging_ = enable;
}

bool StandardGpuResourcesImpl::isInitialized(int device) const {
    // Use default streams as a marker for whether or not a certain
    // device has been initialized
    return defaultStreams_.count(device) != 0;
}

void StandardGpuResourcesImpl::initializeForDevice(int device) {
    if (isInitialized(device)) {
        return;
    }

    FAISS_ASSERT(device < getNumDevices());
    DeviceScope scope(device);

    // If this is the first device that we're initializing, create our
    // pinned memory allocation
    if (defaultStreams_.empty() && pinnedMemSize_ > 0) {
#if defined USE_NVIDIA_CUVS
        // If this is the first device that we're initializing, create our
        // pinned memory allocation
        if (defaultStreams_.empty() && pinnedMemSize_ > 0) {
            try {
                pinnedMemAlloc_ = pmr_->allocate(pinnedMemSize_);
            } catch (const std::bad_alloc& rmm_ex) {
                FAISS_THROW_MSG("CUDA memory allocation error");
            }

            pinnedMemAllocSize_ = pinnedMemSize_;
        }
#else
        auto err = cudaHostAlloc(
                &pinnedMemAlloc_, pinnedMemSize_, cudaHostAllocDefault);

        FAISS_THROW_IF_NOT_FMT(
                err == cudaSuccess,
                "failed to cudaHostAlloc %zu bytes for CPU <-> GPU "
                "async copy buffer (error %d %s)",
                pinnedMemSize_,
                (int)err,
                cudaGetErrorString(err));

        pinnedMemAllocSize_ = pinnedMemSize_;
#endif
    }

    // Make sure that device properties for all devices are cached
    auto& prop = getDeviceProperties(device);

    // Also check to make sure we meet our minimum compute capability (3.0)
    FAISS_ASSERT_FMT(
            prop.major >= 3,
            "Device id %d with CC %d.%d not supported, "
            "need 3.0+ compute capability",
            device,
            prop.major,
            prop.minor);

#if USE_AMD_ROCM
    // Our code is pre-built with and expects warpSize == 32 or 64, validate
    // that
    FAISS_ASSERT_FMT(
            prop.warpSize == 32 || prop.warpSize == 64,
            "Device id %d does not have expected warpSize of 32 or 64",
            device);
#else
    // Our code is pre-built with and expects warpSize == 32, validate that
    FAISS_ASSERT_FMT(
            prop.warpSize == 32,
            "Device id %d does not have expected warpSize of 32",
            device);
#endif

    // Create streams
    cudaStream_t defaultStream = nullptr;
    CUDA_VERIFY(
            cudaStreamCreateWithFlags(&defaultStream, cudaStreamNonBlocking));

    defaultStreams_[device] = defaultStream;

#if defined USE_NVIDIA_CUVS
    raftHandles_.emplace(std::make_pair(device, defaultStream));
#endif

    cudaStream_t asyncCopyStream = nullptr;
    CUDA_VERIFY(
            cudaStreamCreateWithFlags(&asyncCopyStream, cudaStreamNonBlocking));

    asyncCopyStreams_[device] = asyncCopyStream;

    std::vector<cudaStream_t> deviceStreams;
    for (int j = 0; j < kNumStreams; ++j) {
        cudaStream_t stream = nullptr;
        CUDA_VERIFY(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        deviceStreams.push_back(stream);
    }

    alternateStreams_[device] = std::move(deviceStreams);

    // Create cuBLAS handle
    cublasHandle_t blasHandle = nullptr;
    auto blasStatus = cublasCreate(&blasHandle);
    FAISS_ASSERT(blasStatus == CUBLAS_STATUS_SUCCESS);
    blasHandles_[device] = blasHandle;

    // For CUDA 10 on V100, enabling tensor core usage would enable automatic
    // rounding down of inputs to f16 (though accumulate in f32) which results
    // in unacceptable loss of precision in general. For CUDA 11 / A100, only
    // enable tensor core support if it doesn't result in a loss of precision.
#if CUDA_VERSION >= 11000
    cublasSetMathMode(
            blasHandle, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
#endif

    FAISS_ASSERT(allocs_.count(device) == 0);
    allocs_[device] = std::unordered_map<void*, AllocRequest>();

    if (deviceMemSize_ > 0) {
        FAISS_ASSERT(preallocPools_.count(device) == 0);
        preallocPools_.emplace(
                device,
                std::make_unique<PreallocMemoryPool>(
                        device, deviceMemSize_, true /* enableIPC */));
    }

    FAISS_ASSERT(tempMemory_.count(device) == 0);
    auto mem = std::make_unique<StackDeviceMemory>(
            this,
            device,
            // adjust for this specific device
            getDefaultTempMemForGPU(device, tempMemSize_));

    tempMemory_.emplace(device, std::move(mem));
}

cublasHandle_t StandardGpuResourcesImpl::getBlasHandle(int device) {
    initializeForDevice(device);
    return blasHandles_[device];
}

cudaStream_t StandardGpuResourcesImpl::getDefaultStream(int device) {
    initializeForDevice(device);

    auto it = userDefaultStreams_.find(device);
    if (it != userDefaultStreams_.end()) {
        // There is a user override stream set
        return it->second;
    }

    // Otherwise, our base default stream
    return defaultStreams_[device];
}

#if defined USE_NVIDIA_CUVS
raft::device_resources& StandardGpuResourcesImpl::getRaftHandle(int device) {
    initializeForDevice(device);

    auto it = raftHandles_.find(device);
    if (it == raftHandles_.end()) {
        // Make sure we are using the stream the user may have already assigned
        // to the current GpuResources
        raftHandles_.emplace(device, getDefaultStream(device));

        // Initialize cublas handle
        raftHandles_[device].get_cublas_handle();
    }

    // Otherwise, our base default handle
    return raftHandles_[device];
}
#endif

std::vector<cudaStream_t> StandardGpuResourcesImpl::getAlternateStreams(
        int device) {
    initializeForDevice(device);
    return alternateStreams_[device];
}

std::pair<void*, size_t> StandardGpuResourcesImpl::getPinnedMemory() {
    return std::make_pair(pinnedMemAlloc_, pinnedMemAllocSize_);
}

cudaStream_t StandardGpuResourcesImpl::getAsyncCopyStream(int device) {
    initializeForDevice(device);
    return asyncCopyStreams_[device];
}

bool StandardGpuResourcesImpl::requestDeviceMemoryReservationResize(
        int device,
        size_t newSize,
        bool allowShrink) {
    initializeForDevice(device);

    auto it = preallocPools_.find(device);
    if (it == preallocPools_.end() || !it->second) {
        // No preallocated pool for this device
        return false;
    }

    auto* pool = it->second.get();
    size_t currentSize = pool->getTotalSize();

    if (newSize > currentSize) {
        // Expand
        auto result = pool->expand(newSize);
        return result.isSuccess();
    } else if (newSize < currentSize && allowShrink) {
        // Shrink
        auto result = pool->shrink(newSize);
        return result.isSuccess() || result.isPartial();
    }

    // No change needed
    return true;
}

void StandardGpuResourcesImpl::setPreallocPoolIpc(
        const std::string& /* name */,
        size_t /* capacity */) {
    // IPC is handled internally by the PreallocMemoryPool
    // This is a no-op for StandardGpuResourcesImpl as the pool
    // already initializes its own shared memory IPC
}

size_t StandardGpuResourcesImpl::pollPreallocPoolIpc() {
    // IPC polling is handled internally by the PreallocMemoryPool's
    // background thread. This is a no-op that returns 0.
    return 0;
}

std::map<int, PreallocPoolStats> StandardGpuResourcesImpl::getPreallocPoolStats()
        const {
    std::map<int, PreallocPoolStats> stats;

    for (const auto& kv : preallocPools_) {
        if (kv.second) {
            PreallocPoolStats s;
            s.totalBytes = kv.second->getTotalSize();
            s.freeBytes = kv.second->getSizeAvailable();
            s.slabCount = 0;  // Not tracked in current implementation
            s.targetBytes = s.totalBytes;
            s.shrinkPending = false;
            stats[kv.first] = s;
        }
    }

    return stats;
}

void* StandardGpuResourcesImpl::allocMemory(const AllocRequest& req) {
    initializeForDevice(req.device);

    // We don't allocate a placeholder for zero-sized allocations
    if (req.size == 0) {
        return nullptr;
    }

    // cudaMalloc guarantees allocation alignment to 256 bytes; do the same here
    // for alignment purposes (to reduce memory transaction overhead etc)
    auto adjReq = req;
    adjReq.size = utils::roundUp(adjReq.size, (size_t)256);

    void* p = nullptr;
    auto poolIt = preallocPools_.find(adjReq.device);
    bool usePrealloc = (deviceMemSize_ > 0);
    if (usePrealloc) {
        FAISS_ASSERT(poolIt != preallocPools_.end());
    }

    if (adjReq.space == MemorySpace::Temporary) {
        auto& tempMem = tempMemory_[adjReq.device];

        if (adjReq.size > tempMem->getSizeAvailable()) {
            if (usePrealloc) {
                std::stringstream ss;
                ss << "StandardGpuResources: temp alloc fail "
                   << adjReq.toString()
                   << " (no temp space; preallocated pool enforced)\n";
                auto str = ss.str();
                if (allocLogging_) {
                    std::cout << str;
                }
                FAISS_ASSERT_FMT(false, "%s", str.c_str());
            }

            // We need to allocate this ourselves
            AllocRequest newReq = adjReq;
            newReq.space = MemorySpace::Device;
            newReq.type = AllocType::TemporaryMemoryOverflow;

            if (allocLogging_) {
                std::cout
                        << "StandardGpuResources: alloc fail "
                        << adjReq.toString()
                        << " (no temp space); retrying as MemorySpace::Device\n";
            }

            return allocMemory(newReq);
        }

        // Otherwise, we can handle this locally
        p = tempMemory_[adjReq.device]->allocMemory(adjReq.stream, adjReq.size);
    } else if (adjReq.space == MemorySpace::Device) {
        if (usePrealloc) {
            auto* pool = poolIt->second.get();
            FAISS_ASSERT(pool);
            p = pool->allocate(adjReq.size);
            if (!p) {
                std::stringstream ss;
                ss << "StandardGpuResources: device alloc fail "
                   << adjReq.toString()
                   << " (preallocated pool exhausted; available "
                   << pool->getSizeAvailable() << " bytes)\n";
                auto str = ss.str();
                if (allocLogging_) {
                    std::cout << str;
                }
                FAISS_ASSERT_FMT(false, "%s", str.c_str());
            }
        } else {
#if defined USE_NVIDIA_CUVS
            try {
                rmm::mr::device_memory_resource* current_mr =
                        rmm::mr::get_per_device_resource(
                                rmm::cuda_device_id{adjReq.device});
                p = current_mr->allocate_async(adjReq.size, adjReq.stream);
                adjReq.mr = current_mr;
            } catch (const std::bad_alloc& rmm_ex) {
                FAISS_THROW_MSG("CUDA memory allocation error");
            }
#else
            auto err = cudaMalloc(&p, adjReq.size);

            // Throw if we fail to allocate
            if (err != cudaSuccess) {
                // FIXME: as of CUDA 11, a memory allocation error appears to be
                // presented via cudaGetLastError as well, and needs to be
                // cleared. Just call the function to clear it
                cudaGetLastError();

                std::stringstream ss;
                ss << "StandardGpuResources: alloc fail " << adjReq.toString()
                   << " (cudaMalloc error " << cudaGetErrorString(err) << " ["
                   << (int)err << "])\n";
                auto str = ss.str();

                if (allocLogging_) {
                    std::cout << str;
                }

                FAISS_THROW_IF_NOT_FMT(err == cudaSuccess, "%s", str.c_str());
            }
#endif
        }
    } else if (adjReq.space == MemorySpace::Unified) {
        if (usePrealloc) {
            FAISS_THROW_MSG(
                    "Unified memory allocation is not supported when device "
                    "memory reservation is enabled");
        }
#if defined USE_NVIDIA_CUVS
        try {
            // for now, use our own managed MR to do Unified Memory allocations.
            // TODO: change this to use the current device resource once RMM has
            // a way to retrieve a "guaranteed" managed memory resource for a
            // device.
            p = mmr_->allocate_async(adjReq.size, adjReq.stream);
            adjReq.mr = mmr_.get();
        } catch (const std::bad_alloc& rmm_ex) {
            FAISS_THROW_MSG("CUDA memory allocation error");
        }
#else
        auto err = cudaMallocManaged(&p, adjReq.size);

        if (err != cudaSuccess) {
            // FIXME: as of CUDA 11, a memory allocation error appears to be
            // presented via cudaGetLastError as well, and needs to be cleared.
            // Just call the function to clear it
            cudaGetLastError();

            std::stringstream ss;
            ss << "StandardGpuResources: alloc fail " << adjReq.toString()
               << " failed (cudaMallocManaged error " << cudaGetErrorString(err)
               << " [" << (int)err << "])\n";
            auto str = ss.str();

            if (allocLogging_) {
                std::cout << str;
            }

            FAISS_THROW_IF_NOT_FMT(err == cudaSuccess, "%s", str.c_str());
        }
#endif
    } else {
        FAISS_ASSERT_FMT(false, "unknown MemorySpace %d", (int)adjReq.space);
    }

    if (allocLogging_) {
        std::cout << "StandardGpuResources: alloc ok " << adjReq.toString()
                  << " ptr 0x" << p << "\n";
    }

    allocs_[adjReq.device][p] = adjReq;

    return p;
}

void StandardGpuResourcesImpl::deallocMemory(int device, void* p) {
    FAISS_ASSERT(isInitialized(device));

    if (!p) {
        return;
    }

    auto& a = allocs_[device];
    auto it = a.find(p);
    FAISS_ASSERT(it != a.end());

    auto& req = it->second;

    if (allocLogging_) {
        std::cout << "StandardGpuResources: dealloc " << req.toString() << "\n";
    }

    if (req.space == MemorySpace::Temporary) {
        tempMemory_[device]->deallocMemory(device, req.stream, req.size, p);
    } else if (req.space == MemorySpace::Device) {
        if (deviceMemSize_ > 0) {
            auto poolIt = preallocPools_.find(device);
            FAISS_ASSERT(poolIt != preallocPools_.end());
            auto* pool = poolIt->second.get();
            FAISS_ASSERT(pool);
            FAISS_ASSERT_FMT(
                    pool->owns(p),
                    "Pointer does not belong to preallocated pool on device %d",
                    device);
            pool->deallocate(p, req.size);
        } else {
#if defined USE_NVIDIA_CUVS
            req.mr->deallocate_async(p, req.size, req.stream);
#else
            auto err = cudaFree(p);
            FAISS_ASSERT_FMT(
                    err == cudaSuccess,
                    "Failed to cudaFree pointer %p (error %d %s)",
                    p,
                    (int)err,
                    cudaGetErrorString(err));
#endif
        }
    } else if (req.space == MemorySpace::Unified) {
        if (deviceMemSize_ > 0) {
            FAISS_THROW_MSG(
                    "Unified memory deallocation not supported when device "
                    "memory reservation is enabled");
        }
#if defined USE_NVIDIA_CUVS
        req.mr->deallocate_async(p, req.size, req.stream);
#else
        auto err = cudaFree(p);
        FAISS_ASSERT_FMT(
                err == cudaSuccess,
                "Failed to cudaFree pointer %p (error %d %s)",
                p,
                (int)err,
                cudaGetErrorString(err));
#endif
    } else {
        FAISS_ASSERT_FMT(false, "unknown MemorySpace %d", (int)req.space);
    }

    a.erase(it);
}

size_t StandardGpuResourcesImpl::getTempMemoryAvailable(int device) const {
    FAISS_ASSERT(isInitialized(device));

    auto it = tempMemory_.find(device);
    FAISS_ASSERT(it != tempMemory_.end());

    return it->second->getSizeAvailable();
}

std::map<int, std::map<std::string, std::pair<int, size_t>>>
StandardGpuResourcesImpl::getMemoryInfo() const {
    using AT = std::map<std::string, std::pair<int, size_t>>;

    std::map<int, AT> out;

    for (auto& entry : allocs_) {
        AT outDevice;

        for (auto& a : entry.second) {
            auto& v = outDevice[allocTypeToString(a.second.type)];
            v.first++;
            v.second += a.second.size;
        }

        out[entry.first] = std::move(outDevice);
    }

    return out;
}

//
// StandardGpuResources
//

StandardGpuResources::StandardGpuResources()
        : res_(std::make_shared<StandardGpuResourcesImpl>()) {}

StandardGpuResources::~StandardGpuResources() = default;

std::shared_ptr<GpuResources> StandardGpuResources::getResources() {
    return res_;
}

bool StandardGpuResources::supportsBFloat16(int device) {
    return res_->supportsBFloat16(device);
}

bool StandardGpuResources::supportsBFloat16CurrentDevice() {
    return res_->supportsBFloat16CurrentDevice();
}

void StandardGpuResources::noTempMemory() {
    res_->noTempMemory();
}

void StandardGpuResources::setTempMemory(size_t size) {
    res_->setTempMemory(size);
}

void StandardGpuResources::setDeviceMemoryReservation(size_t size) {
    res_->setDeviceMemoryReservation(size);
}

void StandardGpuResources::setPinnedMemory(size_t size) {
    res_->setPinnedMemory(size);
}

void StandardGpuResources::setDefaultStream(int device, cudaStream_t stream) {
    res_->setDefaultStream(device, stream);
}

void StandardGpuResources::revertDefaultStream(int device) {
    res_->revertDefaultStream(device);
}

void StandardGpuResources::setDefaultNullStreamAllDevices() {
    res_->setDefaultNullStreamAllDevices();
}

std::map<int, std::map<std::string, std::pair<int, size_t>>>
StandardGpuResources::getMemoryInfo() const {
    return res_->getMemoryInfo();
}

cudaStream_t StandardGpuResources::getDefaultStream(int device) {
    return res_->getDefaultStream(device);
}

#if defined USE_NVIDIA_CUVS
raft::device_resources& StandardGpuResources::getRaftHandle(int device) {
    return res_->getRaftHandle(device);
}
#endif

size_t StandardGpuResources::getTempMemoryAvailable(int device) const {
    return res_->getTempMemoryAvailable(device);
}

void StandardGpuResources::syncDefaultStreamCurrentDevice() {
    res_->syncDefaultStreamCurrentDevice();
}

void StandardGpuResources::setLogMemoryAllocations(bool enable) {
    res_->setLogMemoryAllocations(enable);
}

} // namespace gpu
} // namespace faiss
