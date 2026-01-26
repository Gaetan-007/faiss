/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/gpu/GpuIndexIVF.h>
#include <faiss/impl/ScalarQuantizer.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace faiss {
struct IndexIVFFlat;
}

namespace faiss {
namespace gpu {

class IVFFlat;
class GpuIndexFlat;

struct GpuIndexIVFFlatConfig : public GpuIndexIVFConfig {
    /// Use the alternative memory layout for the IVF lists
    /// (currently the default)
    bool interleavedLayout = true;
};

/// Wrapper around the GPU implementation that looks like
/// faiss::IndexIVFFlat
class GpuIndexIVFFlat : public GpuIndexIVF {
   public:
    /// Construct from a pre-existing faiss::IndexIVFFlat instance, copying
    /// data over to the given GPU, if the input index is trained.
    GpuIndexIVFFlat(
            GpuResourcesProvider* provider,
            const faiss::IndexIVFFlat* index,
            GpuIndexIVFFlatConfig config = GpuIndexIVFFlatConfig());

    /// Constructs a new instance with an empty flat quantizer; the user
    /// provides the number of IVF lists desired.
    GpuIndexIVFFlat(
            GpuResourcesProvider* provider,
            int dims,
            idx_t nlist,
            faiss::MetricType metric = faiss::METRIC_L2,
            GpuIndexIVFFlatConfig config = GpuIndexIVFFlatConfig());

    /// Constructs a new instance with a provided CPU or GPU coarse quantizer;
    /// the user provides the number of IVF lists desired.
    GpuIndexIVFFlat(
            GpuResourcesProvider* provider,
            Index* coarseQuantizer,
            int dims,
            idx_t nlist,
            faiss::MetricType metric = faiss::METRIC_L2,
            GpuIndexIVFFlatConfig config = GpuIndexIVFFlatConfig());

    ~GpuIndexIVFFlat() override;

    /// Reserve GPU memory in our inverted lists for this number of vectors
    void reserveMemory(size_t numVecs);

    /// Initialize ourselves from the given CPU index; will overwrite
    /// all data in ourselves
    void copyFrom(const faiss::IndexIVFFlat* index);

    /// Copy ourselves to the given CPU index; will overwrite all data
    /// in the index instance
    void copyTo(faiss::IndexIVFFlat* index) const;

    /// After adding vectors, one can call this to reclaim device memory
    /// to exactly the amount needed. Returns space reclaimed in bytes
    size_t reclaimMemory();

    /// Clears out all inverted lists, but retains the coarse centroid
    /// information
    void reset() override;

    /// Should be called if the user ever changes the state of the IVF coarse
    /// quantizer manually (e.g., substitutes a new instance or changes vectors
    /// in the coarse quantizer outside the scope of training)
    void updateQuantizer() override;

    /// Trains the coarse quantizer based on the given vector data
    void train(idx_t n, const float* x) override;

    void reconstruct_n(idx_t i0, idx_t n, float* out) const override;

    /// NOTE: (wangzehao) This function is used to evict a single IVF list (centroid) to CPU memory and free GPU memory
    /// Evict a single IVF list (centroid) to CPU memory and free GPU memory
    size_t evictCentroidToCpu(idx_t listId);

    /// NOTE: (wangzehao) This function is used to load a single IVF list (centroid) from CPU memory back to GPU
    /// Load a single IVF list (centroid) from CPU memory back to GPU
    size_t loadCentroidToGpu(idx_t listId);

    /// NOTE: (wangzehao) This function is used to evict multiple IVF lists (centroids) to CPU memory and free GPU memory
    /// Evict multiple IVF lists (centroids); returns reclaimed bytes per list
    std::vector<uint64_t> evictCentroidsToCpu(
            const std::vector<idx_t>& listIds);

    /// NOTE: (wangzehao) This function is used to load multiple IVF lists (centroids) from CPU memory back to GPU
    /// Load multiple IVF lists (centroids); returns loaded bytes per list
    std::vector<uint64_t> loadCentroidsToGpu(
            const std::vector<idx_t>& listIds);

    /// NOTE: (wangzehao) This function is used to process a shared-memory IPC command; returns true if one was handled
    /// Process a shared-memory IPC command; returns true if one was handled
    bool processSharedMemoryCommand(const char* shmName);

    //
    // Page-fault style auto-fetch management
    // NOTE: (wangzehao) Below functions implement automatic load-on-demand
    // When a search finds that required centroid data is not on GPU,
    // it will automatically fetch from CPU cache (like a page fault handler)
    //

    /// Enable or disable automatic fetching of evicted lists during search
    /// When enabled, search will automatically load missing lists from CPU cache
    void setAutoFetch(bool enable);

    /// Check if auto-fetch is currently enabled
    bool isAutoFetchEnabled() const;

    /// Check if a single IVF list (centroid) is currently on GPU
    bool isListOnGpu(idx_t listId) const;

    /// Get the set of lists that are currently evicted (in CPU cache)
    std::vector<idx_t> getEvictedLists() const;

    /// Get statistics about auto-fetch operations (for debugging/profiling)
    struct AutoFetchStats {
        uint64_t totalFetches;      // Total number of auto-fetch operations
        uint64_t totalListsFetched; // Total number of lists fetched
        uint64_t totalBytesFetched; // Total bytes fetched from CPU
    };
    AutoFetchStats getAutoFetchStats() const;

    /// Reset auto-fetch statistics
    void resetAutoFetchStats();

   protected:
    /// Internal helper: fetch missing lists for a search operation
    /// Returns the number of lists fetched
    size_t fetchMissingListsForSearch_(const std::vector<idx_t>& listIds);

    /// Override searchImpl_ to inject auto-fetch logic before search
    void searchImpl_(
            idx_t n,
            const float* x,
            int k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const override;

   protected:
    /// Initialize appropriate index
    void setIndex_(
            GpuResources* resources,
            int dim,
            int nlist,
            faiss::MetricType metric,
            float metricArg,
            bool useResidual,
            /// Optional ScalarQuantizer
            faiss::ScalarQuantizer* scalarQ,
            bool interleavedLayout,
            IndicesOptions indicesOptions,
            MemorySpace space);

   protected:
    /// Our configuration options
    const GpuIndexIVFFlatConfig ivfFlatConfig_;

    /// Desired inverted list memory reservation
    size_t reserveMemoryVecs_;

    /// Instance that we own; contains the inverted lists
    std::shared_ptr<IVFFlat> index_;

   /// NOTE: (wangzehao) This struct is used to cache the CPU-encoded data and indices of a single IVF list
   private:
    struct CpuListCache {
        std::vector<uint8_t> codes;
        std::vector<idx_t> ids;
    };

    std::unordered_map<idx_t, CpuListCache> cpuListCache_;

    /// NOTE: (wangzehao) Auto-fetch (page-fault style) management members
    /// When true, search will automatically fetch missing lists from CPU
    bool autoFetchEnabled_ = false;

    /// Statistics for auto-fetch operations
    mutable AutoFetchStats autoFetchStats_ = {0, 0, 0};

    static constexpr uint32_t kIpcMagic = 0x4956464c; // "IVFL"
    static constexpr uint32_t kIpcVersion = 1;
    static constexpr uint32_t kIpcStateIdle = 0;
    static constexpr uint32_t kIpcStatePending = 1;
    static constexpr uint32_t kIpcStateDone = 2;
    static constexpr uint32_t kIpcStateError = 3;
    static constexpr uint32_t kIpcOpEvict = 1;
    static constexpr uint32_t kIpcOpLoad = 2;

    struct IpcCommand {
        uint32_t magic;
        uint32_t version;
        std::atomic<uint32_t> state;
        std::atomic<uint32_t> opcode;
        std::atomic<int64_t> listId;
        std::atomic<int64_t> result;
    };
};

} // namespace gpu
} // namespace faiss
