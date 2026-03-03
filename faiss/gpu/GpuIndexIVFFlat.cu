/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/impl/IVFFlat.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/Float16.cuh>
#include <faiss/utils/Heap.h>

#if defined USE_NVIDIA_CUVS
#include <cuvs/neighbors/ivf_flat.hpp>
#include <faiss/gpu/utils/CuvsUtils.h>
#include <faiss/gpu/impl/CuvsIVFFlat.cuh>
#endif

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace faiss {
namespace gpu {

GpuIndexIVFFlat::GpuIndexIVFFlat(
        GpuResourcesProvider* provider,
        const faiss::IndexIVFFlat* index,
        GpuIndexIVFFlatConfig config)
        : GpuIndexIVF(
                  provider,
                  index->d,
                  index->metric_type,
                  index->metric_arg,
                  index->nlist,
                  config),
          ivfFlatConfig_(config),
          reserveMemoryVecs_(0) {
    // Initialize miss policy from config; by default, preserve historical
    // behavior (Error) unless the caller explicitly requests otherwise.
    missPolicy_ = ivfFlatConfig_.missPolicy;
    if (missPolicy_ == IvfListMissPolicy::AutoFetch) {
        autoFetchEnabled_ = true;
    }
    copyFrom(index);
}

GpuIndexIVFFlat::GpuIndexIVFFlat(
        GpuResourcesProvider* provider,
        int dims,
        idx_t nlist,
        faiss::MetricType metric,
        GpuIndexIVFFlatConfig config)
        : GpuIndexIVF(provider, dims, metric, 0, nlist, config),
          ivfFlatConfig_(config),
          reserveMemoryVecs_(0) {
    missPolicy_ = ivfFlatConfig_.missPolicy;
    if (missPolicy_ == IvfListMissPolicy::AutoFetch) {
        autoFetchEnabled_ = true;
    }
    // We haven't trained ourselves, so don't construct the IVFFlat
    // index yet
}

GpuIndexIVFFlat::GpuIndexIVFFlat(
        GpuResourcesProvider* provider,
        Index* coarseQuantizer,
        int dims,
        idx_t nlist,
        faiss::MetricType metric,
        GpuIndexIVFFlatConfig config)
        : GpuIndexIVF(
                  provider,
                  coarseQuantizer,
                  dims,
                  metric,
                  0,
                  nlist,
                  config),
          ivfFlatConfig_(config),
          reserveMemoryVecs_(0) {
    missPolicy_ = ivfFlatConfig_.missPolicy;
    if (missPolicy_ == IvfListMissPolicy::AutoFetch) {
        autoFetchEnabled_ = true;
    }
    // We could have been passed an already trained coarse quantizer. There is
    // no other quantizer that we need to train, so this is sufficient
    if (this->is_trained) {
        FAISS_ASSERT(this->quantizer);
        setIndex_(
                resources_.get(),
                this->d,
                this->nlist,
                this->metric_type,
                this->metric_arg,
                false,   // no residual
                nullptr, // no scalar quantizer
                ivfFlatConfig_.interleavedLayout,
                ivfFlatConfig_.indicesOptions,
                config_.memorySpace);
        baseIndex_ = std::static_pointer_cast<IVFBase, IVFFlat>(index_);
        updateQuantizer();
    }
}

GpuIndexIVFFlat::~GpuIndexIVFFlat() {}

void GpuIndexIVFFlat::reserveMemory(size_t numVecs) {
    DeviceScope scope(config_.device);

    if (should_use_cuvs(config_)) {
        FAISS_THROW_MSG(
                "Pre-allocation of IVF lists is not supported with cuVS enabled.");
    }

    reserveMemoryVecs_ = numVecs;
    if (index_) {
        index_->reserveMemory(numVecs);
    }
}

void GpuIndexIVFFlat::copyFrom(const faiss::IndexIVFFlat* index) {
    DeviceScope scope(config_.device);

    // This will copy GpuIndexIVF data such as the coarse quantizer
    GpuIndexIVF::copyFrom(index);

    // Clear out our old data
    index_.reset();
    cpuListCache_.clear();
    externalIndex_ = nullptr;

    // skip base class allocations if cuVS is enabled
    if (!should_use_cuvs(config_)) {
        baseIndex_.reset();
    }

    // The other index might not be trained
    if (!index->is_trained) {
        FAISS_ASSERT(!is_trained);
        return;
    }

    // Otherwise, we can populate ourselves from the other index
    FAISS_ASSERT(is_trained);

    // Copy our lists as well
    setIndex_(
            resources_.get(),
            d,
            nlist,
            index->metric_type,
            index->metric_arg,
            false,   // no residual
            nullptr, // no scalar quantizer
            ivfFlatConfig_.interleavedLayout,
            ivfFlatConfig_.indicesOptions,
            config_.memorySpace);
    baseIndex_ = std::static_pointer_cast<IVFBase, IVFFlat>(index_);
    updateQuantizer();

    // Copy all of the IVF data
    index_->copyInvertedListsFrom(index->invlists);
}

void GpuIndexIVFFlat::copyFromSelective(
        const faiss::IndexIVFFlat* index,
        const std::vector<idx_t>& listIds) {
    DeviceScope scope(config_.device);

    FAISS_THROW_IF_NOT_MSG(index, "copyFromSelective: index must not be null");

    // Record external CPU backing index for later no-copy eviction and loads.
    externalIndex_ = index;

    // This will copy GpuIndexIVF data such as the coarse quantizer
    GpuIndexIVF::copyFrom(index);

    // Clear out our old data
    index_.reset();

    // skip base class allocations if cuVS is enabled
    if (!should_use_cuvs(config_)) {
        baseIndex_.reset();
    }
    cpuListCache_.clear();

    // The other index might not be trained
    if (!index->is_trained) {
        FAISS_ASSERT(!is_trained);
        return;
    }

    // Otherwise, we can populate ourselves from the other index
    FAISS_ASSERT(is_trained);

    // Copy our lists as well
    setIndex_(
            resources_.get(),
            d,
            nlist,
            index->metric_type,
            index->metric_arg,
            false,   // no residual
            nullptr, // no scalar quantizer
            ivfFlatConfig_.interleavedLayout,
            ivfFlatConfig_.indicesOptions,
            config_.memorySpace);
    baseIndex_ = std::static_pointer_cast<IVFBase, IVFFlat>(index_);
    updateQuantizer();

    auto* invlists = index->invlists;
    FAISS_THROW_IF_NOT_MSG(
            invlists, "copyFromSelective: source index has no invlists");
    FAISS_THROW_IF_NOT_MSG(
            invlists->nlist == nlist,
            "copyFromSelective: invlists nlist mismatch");

    std::unordered_set<idx_t> listsToLoad;
    listsToLoad.reserve(listIds.size());
    for (auto listId : listIds) {
        FAISS_THROW_IF_NOT_FMT(
                listId >= 0 && listId < nlist,
                "copyFromSelective: list %ld out of bounds (%ld lists total)",
                listId,
                nlist);
        listsToLoad.insert(listId);
    }

    for (idx_t listId = 0; listId < nlist; ++listId) {
        auto numVecs = invlists->list_size(listId);
        if (numVecs == 0) {
            // No data to load or cache
            continue;
        }

        const auto* codes =
                reinterpret_cast<const uint8_t*>(invlists->get_codes(listId));
        const auto* ids = invlists->get_ids(listId);

        FAISS_THROW_IF_NOT_MSG(
                codes,
                "copyFromSelective: null codes for a non-empty list");
        if (ivfFlatConfig_.indicesOptions != INDICES_IVF) {
            FAISS_THROW_IF_NOT_MSG(
                    ids,
                    "copyFromSelective: null ids for a non-empty list");
        }

        if (listsToLoad.count(listId) > 0) {
            index_->addEncodedVectorsToListFromCpu(
                    listId,
                    codes,
                    ids,
                    numVecs);
        } else {
            CpuListCache cache;
            cache.backingType = CpuCacheBackingType::ExternalInvlists;
            cache.state = CpuCacheState::Clean;
            cache.externalIndex = index;
            cache.externalListId = listId;
            cpuListCache_.emplace(listId, std::move(cache));
        }
    }
}

void GpuIndexIVFFlat::copyTo(faiss::IndexIVFFlat* index) const {
    DeviceScope scope(config_.device);

    // We must have the indices in order to copy to ourselves
    FAISS_THROW_IF_NOT_MSG(
            ivfFlatConfig_.indicesOptions != INDICES_IVF,
            "Cannot copy to CPU as GPU index doesn't retain "
            "indices (INDICES_IVF)");

    GpuIndexIVF::copyTo(index);
    index->code_size = this->d * sizeof(float);

    auto ivf = new ArrayInvertedLists(nlist, index->code_size);
    index->replace_invlists(ivf, true);

    if (index_) {
        // Copy IVF lists
        index_->copyInvertedListsTo(ivf);
    }
}

size_t GpuIndexIVFFlat::reclaimMemory() {
    DeviceScope scope(config_.device);

    if (index_) {
        return index_->reclaimMemory();
    }

    return 0;
}

void GpuIndexIVFFlat::reset() {
    DeviceScope scope(config_.device);

    if (index_) {
        index_->reset();
        this->ntotal = 0;
    } else {
        FAISS_ASSERT(this->ntotal == 0);
    }
}

void GpuIndexIVFFlat::updateQuantizer() {
    FAISS_THROW_IF_NOT_MSG(
            quantizer, "Calling updateQuantizer without a quantizer instance");

    // Only need to do something if we are already initialized
    if (index_) {
        index_->updateQuantizer(quantizer);
    }
}

/// NOTE: (wangzehao) This function is used to evict a single IVF list (centroid) to CPU memory and free GPU memory
size_t GpuIndexIVFFlat::evictCentroidToCpu(idx_t listId) {
    DeviceScope scope(config_.device);

    FAISS_THROW_IF_NOT_MSG(index_, "IVF index not initialized");
    FAISS_THROW_IF_NOT_FMT(
            listId < nlist,
            "IVF list %ld is out of bounds (%ld lists total)",
            listId,
            nlist);

    auto it = cpuListCache_.find(listId);
    if (it != cpuListCache_.end()) {
        // Already known to be backed by CPU (either internal copy or external);
        // nothing to do.
        return 0;
    }

    // If no-copy eviction is enabled and we have an external CPU backing
    // index, avoid issuing a GPU->CPU copy and rely on that backing instead.
    if (allowNoCopyEvict_) {
        FAISS_THROW_IF_NOT_MSG(
                externalIndex_,
                "GpuIndexIVFFlat::evictCentroidToCpu: no external backing "
                "IndexIVFFlat configured for no-copy eviction");

        auto* invlists = externalIndex_->invlists;
        FAISS_THROW_IF_NOT_MSG(
                invlists,
                "GpuIndexIVFFlat::evictCentroidToCpu: external backing index "
                "has no invlists");

        FAISS_THROW_IF_NOT_FMT(
                listId >= 0 && listId < invlists->nlist,
                "GpuIndexIVFFlat::evictCentroidToCpu: external list %ld out "
                "of bounds (%ld lists total)",
                listId,
                invlists->nlist);

        // Optionally validate list lengths when possible.
        if (baseIndex_) {
            auto cpuLen = invlists->list_size(listId);
            auto gpuLen = baseIndex_->getListLength(listId);
            FAISS_THROW_IF_NOT_FMT(
                    cpuLen == gpuLen,
                    "GpuIndexIVFFlat::evictCentroidToCpu: external list %ld "
                    "length mismatch (CPU %ld, GPU %ld)",
                    listId,
                    cpuLen,
                    gpuLen);
        }

        CpuListCache cache;
        cache.backingType = CpuCacheBackingType::ExternalInvlists;
        cache.state = CpuCacheState::Clean;
        cache.externalIndex = externalIndex_;
        cache.externalListId = listId;
        cpuListCache_.emplace(listId, std::move(cache));

        // Ensure any pending operations on the IVF lists complete before
        // we reclaim device memory.
        resources_->syncDefaultStream(config_.device);
        return index_->evictList(listId);
    }

    // Fallback: perform a defensive GPU->CPU copy into an internal cache
    // entry before evicting.
    CpuListCache cache;
    cache.backingType = CpuCacheBackingType::InternalCopy;
    cache.state = CpuCacheState::Clean;
    cache.codes = index_->getListVectorData(listId, false);
    cache.ids = index_->getListIndices(listId);
    cpuListCache_.emplace(listId, std::move(cache));

    // Sync before evictList: copyToHost uses cudaMemcpyAsync; we must ensure
    // the copy completes before freeing device memory (prevents use-after-free
    // and pool corruption when setDeviceMemoryReservation is used).
    resources_->syncDefaultStream(config_.device);

    return index_->evictList(listId);
}

/// NOTE: (wangzehao) This function is used to load a single IVF list (centroid) from CPU memory back to GPU
size_t GpuIndexIVFFlat::loadCentroidToGpu(idx_t listId) {
    DeviceScope scope(config_.device);

    FAISS_THROW_IF_NOT_MSG(index_, "IVF index not initialized");
    FAISS_THROW_IF_NOT_FMT(
            listId < nlist,
            "IVF list %ld is out of bounds (%ld lists total)",
            listId,
            nlist);

    auto it = cpuListCache_.find(listId);
    if (it == cpuListCache_.end()) {
        return 0;
    }

    auto& cache = it->second;
    size_t bytesLoaded = 0;

    if (cache.backingType == CpuCacheBackingType::InternalCopy) {
        auto numVecs = (idx_t)cache.ids.size();
        if (numVecs == 0 && !cache.codes.empty()) {
            numVecs = (idx_t)(cache.codes.size() / (this->d * sizeof(float)));
        }

        if (ivfFlatConfig_.indicesOptions != INDICES_IVF && numVecs > 0) {
            FAISS_THROW_IF_NOT_MSG(
                    !cache.ids.empty(),
                    "Cached IVF list is missing indices for GPU load");
        }

        index_->addEncodedVectorsToListFromCpu(
                listId,
                cache.codes.data(),
                cache.ids.empty() ? nullptr : cache.ids.data(),
                numVecs);

        bytesLoaded =
                cache.codes.size() + cache.ids.size() * sizeof(idx_t);
    } else if (cache.backingType == CpuCacheBackingType::ExternalInvlists) {
        FAISS_THROW_IF_NOT_MSG(
                cache.externalIndex,
                "loadCentroidToGpu: external-backed cache entry has no "
                "source IndexIVFFlat");

        auto* invlists = cache.externalIndex->invlists;
        FAISS_THROW_IF_NOT_MSG(
                invlists,
                "loadCentroidToGpu: external backing index has no invlists");

        FAISS_THROW_IF_NOT_FMT(
                cache.externalListId >= 0 &&
                        cache.externalListId < invlists->nlist,
                "loadCentroidToGpu: external list %ld out of bounds (%ld "
                "lists total)",
                cache.externalListId,
                invlists->nlist);

        auto numVecs = invlists->list_size(cache.externalListId);
        if (numVecs == 0) {
            cpuListCache_.erase(it);
            return 0;
        }

        const auto* codes = reinterpret_cast<const uint8_t*>(
                invlists->get_codes(cache.externalListId));
        const auto* ids = invlists->get_ids(cache.externalListId);

        FAISS_THROW_IF_NOT_MSG(
                codes,
                "loadCentroidToGpu: external-backed IVF list has null codes");
        if (ivfFlatConfig_.indicesOptions != INDICES_IVF && numVecs > 0) {
            FAISS_THROW_IF_NOT_MSG(
                    ids,
                    "loadCentroidToGpu: external-backed IVF list is missing "
                    "indices");
        }

        index_->addEncodedVectorsToListFromCpu(listId, codes, ids, numVecs);

        bytesLoaded = static_cast<size_t>(numVecs) *
                static_cast<size_t>(cache.externalIndex->code_size);
        if (ids) {
            bytesLoaded += static_cast<size_t>(numVecs) * sizeof(idx_t);
        }
    } else {
        FAISS_THROW_MSG(
                "loadCentroidToGpu: unknown CpuCacheBackingType for IVF list");
    }

    cpuListCache_.erase(it);

    return bytesLoaded;
}

/// NOTE: (wangzehao) This function is used to evict multiple IVF lists (centroids) to CPU memory and free GPU memory
std::vector<uint64_t> GpuIndexIVFFlat::evictCentroidsToCpu(
        const std::vector<idx_t>& listIds) {
    std::vector<uint64_t> out;
    out.reserve(listIds.size());

    for (auto listId : listIds) {
        out.push_back(static_cast<uint64_t>(evictCentroidToCpu(listId)));
    }

    return out;
}

/// NOTE: (wangzehao) This function is used to load multiple IVF lists (centroids) from CPU memory back to GPU
std::vector<uint64_t> GpuIndexIVFFlat::loadCentroidsToGpu(
        const std::vector<idx_t>& listIds) {
    std::vector<uint64_t> out;
    out.reserve(listIds.size());

    for (auto listId : listIds) {
        out.push_back(static_cast<uint64_t>(loadCentroidToGpu(listId)));
    }

    return out;
}

/// NOTE: (wangzehao) This function is used to process a shared-memory IPC command; returns true if one was handled
bool GpuIndexIVFFlat::processSharedMemoryCommand(const char* shmName) {
    FAISS_THROW_IF_NOT_MSG(shmName, "shmName must not be null");

    int fd = shm_open(shmName, O_RDWR, 0666);
    if (fd < 0) {
        return false;
    }

    auto* cmd = static_cast<IpcCommand*>(mmap(
            nullptr,
            sizeof(IpcCommand),
            PROT_READ | PROT_WRITE,
            MAP_SHARED,
            fd,
            0));
    close(fd);

    if (cmd == MAP_FAILED) {
        return false;
    }

    bool handled = false;

    if (cmd->magic == kIpcMagic && cmd->version == kIpcVersion &&
        cmd->state.load(std::memory_order_acquire) == kIpcStatePending) {
        handled = true;
        cmd->result.store(0, std::memory_order_release);

        try {
            auto op = cmd->opcode.load(std::memory_order_acquire);
            auto listId = static_cast<idx_t>(
                    cmd->listId.load(std::memory_order_acquire));

            size_t result = 0;
            if (op == kIpcOpEvict) {
                result = evictCentroidToCpu(listId);
                cmd->state.store(kIpcStateDone, std::memory_order_release);
            } else if (op == kIpcOpLoad) {
                result = loadCentroidToGpu(listId);
                cmd->state.store(kIpcStateDone, std::memory_order_release);
            } else {
                cmd->state.store(kIpcStateError, std::memory_order_release);
                cmd->result.store(-1, std::memory_order_release);
            }

            if (op == kIpcOpEvict || op == kIpcOpLoad) {
                cmd->result.store(
                        static_cast<int64_t>(result),
                        std::memory_order_release);
            }
        } catch (const std::exception&) {
            cmd->state.store(kIpcStateError, std::memory_order_release);
            cmd->result.store(-1, std::memory_order_release);
        }
    }

    munmap(cmd, sizeof(IpcCommand));
    return handled;
}

//
// Page-fault style auto-fetch implementation
// NOTE: (wangzehao) Below functions implement automatic load-on-demand
//

void GpuIndexIVFFlat::setAutoFetch(bool enable) {
    autoFetchEnabled_ = enable;
    if (enable) {
        // Align the higher-level policy with the legacy auto-fetch flag
        missPolicy_ = IvfListMissPolicy::AutoFetch;
    } else if (missPolicy_ == IvfListMissPolicy::AutoFetch) {
        // Revert to the historical default behavior when disabling auto-fetch
        missPolicy_ = IvfListMissPolicy::Error;
    }
}

bool GpuIndexIVFFlat::isAutoFetchEnabled() const {
    // Keep this in sync with the miss policy; callers that only know about
    // the legacy API still see consistent behavior.
    return missPolicy_ == IvfListMissPolicy::AutoFetch;
}

void GpuIndexIVFFlat::setMissPolicy(IvfListMissPolicy policy) {
    missPolicy_ = policy;

    // Auto-fetch flag is an implementation detail of the AutoFetch policy.
    if (policy == IvfListMissPolicy::AutoFetch) {
        autoFetchEnabled_ = true;
    } else {
        autoFetchEnabled_ = false;
    }
}

IvfListMissPolicy GpuIndexIVFFlat::getMissPolicy() const {
    return missPolicy_;
}

void GpuIndexIVFFlat::setNoCopyEvictEnabled(bool enable) {
    if (enable) {
        FAISS_THROW_IF_NOT_MSG(
                externalIndex_,
                "GpuIndexIVFFlat::setNoCopyEvictEnabled: no external "
                "IndexIVFFlat backing configured; build this index with "
                "copyFromSelective or provide an external backing index "
                "before enabling no-copy eviction");
    }
    allowNoCopyEvict_ = enable;
}

bool GpuIndexIVFFlat::isNoCopyEvictEnabled() const {
    return allowNoCopyEvict_;
}

bool GpuIndexIVFFlat::isListOnGpu(idx_t listId) const {
    DeviceScope scope(config_.device);

    FAISS_THROW_IF_NOT_MSG(index_, "IVF index not initialized");
    FAISS_THROW_IF_NOT_FMT(
            listId < nlist,
            "IVF list %ld is out of bounds (%ld lists total)",
            listId,
            nlist);

    return index_->isListOnGpu(listId);
}

std::vector<idx_t> GpuIndexIVFFlat::getEvictedLists() const {
    DeviceScope scope(config_.device);

    std::vector<idx_t> evicted;
    evicted.reserve(cpuListCache_.size());

    for (const auto& kv : cpuListCache_) {
        evicted.push_back(kv.first);
    }

    return evicted;
}

GpuIndexIVFFlat::AutoFetchStats GpuIndexIVFFlat::getAutoFetchStats() const {
    return autoFetchStats_;
}

void GpuIndexIVFFlat::resetAutoFetchStats() {
    autoFetchStats_ = {0, 0, 0};
}

size_t GpuIndexIVFFlat::fetchMissingListsForSearch_(
        const std::vector<idx_t>& listIds) {
    if (!autoFetchEnabled_) {
        return 0;
    }

    // Find which lists are in the CPU cache (i.e., evicted from GPU)
    std::vector<idx_t> toFetch;
    toFetch.reserve(listIds.size());

    for (auto listId : listIds) {
        if (listId < 0 || listId >= nlist) {
            continue;
        }
        // Check if this list is in CPU cache (meaning it was evicted)
        if (cpuListCache_.count(listId) > 0) {
            toFetch.push_back(listId);
        }
    }

    if (toFetch.empty()) {
        return 0;
    }

    // Load missing lists from CPU cache to GPU
    size_t totalBytes = 0;
    for (auto listId : toFetch) {
        size_t bytes = loadCentroidToGpu(listId);
        totalBytes += bytes;
    }

    // Update statistics
    autoFetchStats_.totalFetches++;
    autoFetchStats_.totalListsFetched += toFetch.size();
    autoFetchStats_.totalBytesFetched += totalBytes;

    return toFetch.size();
}

void GpuIndexIVFFlat::searchImpl_(
        idx_t n,
        const float* x,
        int k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    // Device should already be set by GpuIndex::search
    DeviceScope scope(config_.device);

    // Fast path: if no special miss policy or no CPU-side cached lists,
    // delegate to the base IVF implementation.
    if (missPolicy_ == IvfListMissPolicy::Error || cpuListCache_.empty()) {
        if (missPolicy_ == IvfListMissPolicy::AutoFetch &&
            !cpuListCache_.empty()) {
            // fall through to AutoFetch handling below
        } else {
            GpuIndexIVF::searchImpl_(n, x, k, distances, labels, params);
            return;
        }
    }

    auto stream = resources_->getDefaultStream(config_.device);

    // Existing AutoFetch behavior: pre-load any missing lists to GPU, then
    // delegate to the base implementation.
    if (missPolicy_ == IvfListMissPolicy::AutoFetch) {
        if (cpuListCache_.empty()) {
            GpuIndexIVF::searchImpl_(n, x, k, distances, labels, params);
            return;
        }

        int use_nprobe = getCurrentNProbe_(params);

        // Allocate space for coarse quantizer results
        std::vector<idx_t> coarseIndices(n * use_nprobe);
        std::vector<float> coarseDistances(n * use_nprobe);

        // Perform coarse quantizer search to get which lists we need
        quantizer->search(
                n,
                x,
                use_nprobe,
                coarseDistances.data(),
                coarseIndices.data());

        // Deduplicate the list IDs we need to access
        std::unordered_set<idx_t> uniqueListIds(
                coarseIndices.begin(), coarseIndices.end());
        std::vector<idx_t> listIdsToCheck(
                uniqueListIds.begin(), uniqueListIds.end());

        // Fetch any missing lists from CPU cache
        // Note: we cast away const here because auto-fetch modifies internal
        // state. This is acceptable because the logical result of the search
        // does not change.
        const_cast<GpuIndexIVFFlat*>(this)->fetchMissingListsForSearch_(
                listIdsToCheck);

        // Now perform the actual search
        GpuIndexIVF::searchImpl_(n, x, k, distances, labels, params);
        return;
    }

    // New CpuOffload behavior: perform a joint GPU + CPU search where probed
    // lists that are resident on GPU are searched on GPU, and probed lists
    // that have been evicted are searched on CPU, with results merged on CPU.
    if (missPolicy_ == IvfListMissPolicy::CpuOffload) {
        // If there are no cached lists, just run a pure-GPU search.
        if (cpuListCache_.empty()) {
            GpuIndexIVF::searchImpl_(n, x, k, distances, labels, params);
            return;
        }

        FAISS_THROW_IF_NOT_MSG(
                externalIndex_,
                "GpuIndexIVFFlat::searchImpl_: CpuOffload policy requires an "
                "external CPU IndexIVFFlat backing index (configure via "
                "copyFromSelective or equivalent)");

        // Determine how many probes we should use.
        int use_nprobe = getCurrentNProbe_(params);
        FAISS_THROW_IF_NOT(use_nprobe > 0);

        // Copy queries back to host once so they can be reused by both the
        // GPU preassigned path and the CPU IndexIVFFlat search.
        auto hostQueries = toHost<float, 2>(
                const_cast<float*>(x), stream, {n, this->d});

        // Coarse quantization on the host queries to obtain list assignments
        // and distances to IVF centroids.
        std::vector<idx_t> coarseIndices(n * use_nprobe);
        std::vector<float> coarseDistances(n * use_nprobe);

        quantizer->search(
                n,
                hostQueries.data(),
                use_nprobe,
                coarseDistances.data(),
                coarseIndices.data());

        // Partition assignments into GPU-handled and CPU-handled lists.
        size_t totalProbes = (size_t)n * (size_t)use_nprobe;
        std::vector<idx_t> assignGpu(totalProbes, (idx_t)-1);
        std::vector<idx_t> assignCpu(totalProbes, (idx_t)-1);

        bool hasGpuLists = false;
        bool hasCpuLists = false;

        for (idx_t i = 0; i < n; ++i) {
            for (int p = 0; p < use_nprobe; ++p) {
                size_t pos = (size_t)i * (size_t)use_nprobe + (size_t)p;
                idx_t key = coarseIndices[pos];

                if (key < 0) {
                    continue;
                }

                FAISS_THROW_IF_NOT_FMT(
                        key < nlist,
                        "GpuIndexIVFFlat::searchImpl_: invalid IVF list id %ld "
                        "(nlist %ld)",
                        key,
                        nlist);

                bool onGpu = index_ && index_->isListOnGpu(key);
                auto cpuIt = cpuListCache_.find(key);
                bool inCpuCache = cpuIt != cpuListCache_.end();

                if (onGpu) {
                    assignGpu[pos] = key;
                    hasGpuLists = true;
                } else if (inCpuCache) {
                    assignCpu[pos] = key;
                    hasCpuLists = true;
                } else {
                    FAISS_THROW_IF_NOT_MSG(
                            false,
                            "GpuIndexIVFFlat::searchImpl_: CpuOffload policy "
                            "requires probed lists to be either resident on "
                            "GPU or present in CPU cache / external backing");
                }
            }
        }

        // Prepare IVF search parameters for both GPU and CPU preassigned search
        IVFSearchParameters baseParams;
        const IVFSearchParameters* userParams =
                dynamic_cast<const IVFSearchParameters*>(params);
        if (userParams) {
            baseParams = *userParams;
        }
        baseParams.nprobe = (size_t)use_nprobe;
        baseParams.max_codes = 0;

        // GPU partial results
        std::vector<float> D_gpu;
        std::vector<idx_t> I_gpu;

        if (hasGpuLists) {
            D_gpu.resize((size_t)n * (size_t)k);
            I_gpu.resize((size_t)n * (size_t)k);

            // Use the GPU preassigned search path restricted to GPU-resident
            // lists.
            this->search_preassigned(
                    n,
                    hostQueries.data(),
                    k,
                    assignGpu.data(),
                    coarseDistances.data(),
                    D_gpu.data(),
                    I_gpu.data(),
                    /*store_pairs=*/false,
                    &baseParams,
                    /*stats=*/nullptr);
        }

        // CPU partial results
        std::vector<float> D_cpu;
        std::vector<idx_t> I_cpu;

        if (hasCpuLists) {
            D_cpu.resize((size_t)n * (size_t)k);
            I_cpu.resize((size_t)n * (size_t)k);

            externalIndex_->search_preassigned(
                    n,
                    hostQueries.data(),
                    k,
                    assignCpu.data(),
                    coarseDistances.data(),
                    D_cpu.data(),
                    I_cpu.data(),
                    /*store_pairs=*/false,
                    &baseParams,
                    /*stats=*/nullptr);
        }

        // Merge GPU and CPU results per query on the CPU.
        std::vector<float> finalDistances((size_t)n * (size_t)k);
        std::vector<idx_t> finalLabels((size_t)n * (size_t)k);

        if (hasGpuLists && hasCpuLists) {
            // Two-way merge between GPU and CPU shards.
            std::vector<float> allDistances(2 * (size_t)n * (size_t)k);
            std::vector<idx_t> allLabels(2 * (size_t)n * (size_t)k);

            size_t blockSize = (size_t)n * (size_t)k;
            std::copy(D_gpu.begin(), D_gpu.end(), allDistances.begin());
            std::copy(
                    D_cpu.begin(),
                    D_cpu.end(),
                    allDistances.begin() + blockSize);

            std::copy(I_gpu.begin(), I_gpu.end(), allLabels.begin());
            std::copy(
                    I_cpu.begin(),
                    I_cpu.end(),
                    allLabels.begin() + blockSize);

            if (metric_type == METRIC_L2) {
                merge_knn_results<idx_t, CMin<float, int>>(
                        (size_t)n,
                        (size_t)k,
                        2,
                        allDistances.data(),
                        allLabels.data(),
                        finalDistances.data(),
                        finalLabels.data());
            } else {
                merge_knn_results<idx_t, CMax<float, int>>(
                        (size_t)n,
                        (size_t)k,
                        2,
                        allDistances.data(),
                        allLabels.data(),
                        finalDistances.data(),
                        finalLabels.data());
            }
        } else if (hasGpuLists) {
            finalDistances = std::move(D_gpu);
            finalLabels = std::move(I_gpu);
        } else if (hasCpuLists) {
            finalDistances = std::move(D_cpu);
            finalLabels = std::move(I_cpu);
        } else {
            // No valid lists at all (all assignments < 0); fill with neutral
            // values.
            float neutral =
                    (metric_type == METRIC_L2)
                    ? std::numeric_limits<float>::infinity()
                    : -std::numeric_limits<float>::infinity();
            std::fill(finalDistances.begin(), finalDistances.end(), neutral);
            std::fill(finalLabels.begin(), finalLabels.end(), idx_t(-1));
        }

        // Copy merged results back to the device buffers expected by
        // GpuIndex::search, so that the higher-level code can copy them back
        // to the user-provided host arrays.
        size_t numVals = (size_t)n * (size_t)k;
        CUDA_VERIFY(cudaMemcpyAsync(
                distances,
                finalDistances.data(),
                numVals * sizeof(float),
                cudaMemcpyHostToDevice,
                stream));
        CUDA_VERIFY(cudaMemcpyAsync(
                labels,
                finalLabels.data(),
                numVals * sizeof(idx_t),
                cudaMemcpyHostToDevice,
                stream));

        return;
    }

    // Fallback safety: if we reach here with an unknown policy value, just
    // delegate to the base IVF implementation.
    GpuIndexIVF::searchImpl_(n, x, k, distances, labels, params);
}

void GpuIndexIVFFlat::addImpl_(idx_t n, const float* x, const idx_t* ids) {
    // When no-copy eviction with external CPU backing is enabled, we require
    // the IVF lists to remain consistent with the external source. For now,
    // enforce a read-only policy in this mode to avoid silent divergence.
    if (allowNoCopyEvict_ && externalIndex_) {
        FAISS_THROW_MSG(
                "GpuIndexIVFFlat::addImpl_: adding vectors is not allowed "
                "while no-copy eviction with external CPU backing is "
                "enabled; build a new GPU index or disable no-copy eviction");
    }

    GpuIndexIVF::addImpl_(n, x, ids);
}

void GpuIndexIVFFlat::train(idx_t n, const float* x) {
    DeviceScope scope(config_.device);

    // just in case someone changed our quantizer
    verifyIVFSettings_();

    if (this->is_trained) {
        FAISS_ASSERT(index_);
        if (should_use_cuvs(config_)) {
            // copy the IVF centroids to the cuVS index
            // in case it has been reset. This is because `reset` clears the
            // cuVS index and its centroids.
            // TODO: change this once the coarse quantizer is separated from
            // cuVS index
            updateQuantizer();
        };
        return;
    }

    FAISS_ASSERT(!index_);

    if (should_use_cuvs(config_)) {
#if defined USE_NVIDIA_CUVS
        setIndex_(
                resources_.get(),
                this->d,
                this->nlist,
                this->metric_type,
                this->metric_arg,
                false,   // no residual
                nullptr, // no scalar quantizer
                ivfFlatConfig_.interleavedLayout,
                ivfFlatConfig_.indicesOptions,
                config_.memorySpace);
        const raft::device_resources& raft_handle =
                resources_->getRaftHandleCurrentDevice();

        cuvs::neighbors::ivf_flat::index_params cuvs_index_params;
        cuvs_index_params.n_lists = nlist;
        cuvs_index_params.metric = metricFaissToCuvs(metric_type, false);
        cuvs_index_params.add_data_on_build = false;
        cuvs_index_params.kmeans_trainset_fraction =
                static_cast<double>(cp.max_points_per_centroid * nlist) /
                static_cast<double>(n);
        cuvs_index_params.kmeans_n_iters = cp.niter;

        auto cuvsIndex_ =
                std::static_pointer_cast<CuvsIVFFlat, IVFFlat>(index_);

        std::optional<cuvs::neighbors::ivf_flat::index<float, idx_t>>
                cuvs_ivfflat_index;

        if (getDeviceForAddress(x) >= 0) {
            auto dataset_d =
                    raft::make_device_matrix_view<const float, idx_t>(x, n, d);
            cuvs_ivfflat_index = cuvs::neighbors::ivf_flat::build(
                    raft_handle, cuvs_index_params, dataset_d);
        } else {
            auto dataset_h =
                    raft::make_host_matrix_view<const float, idx_t>(x, n, d);
            cuvs_ivfflat_index = cuvs::neighbors::ivf_flat::build(
                    raft_handle, cuvs_index_params, dataset_h);
        }

        if (isGpuIndex(quantizer)) {
            quantizer->train(
                    nlist, cuvs_ivfflat_index.value().centers().data_handle());
            quantizer->add(
                    nlist, cuvs_ivfflat_index.value().centers().data_handle());
        } else {
            // transfer centroids to host
            auto host_centroids = toHost<float, 2>(
                    cuvs_ivfflat_index.value().centers().data_handle(),
                    raft_handle.get_stream(),
                    {idx_t(nlist), this->d});
            quantizer->train(nlist, host_centroids.data());
            quantizer->add(nlist, host_centroids.data());
        }

        cuvsIndex_->setCuvsIndex(std::move(*cuvs_ivfflat_index));
#else
        FAISS_THROW_MSG(
                "cuVS has not been compiled into the current version so it cannot be used.");
#endif
    } else {
        // FIXME: GPUize more of this
        // First, make sure that the data is resident on the CPU, if it is not
        // on the CPU, as we depend upon parts of the CPU code
        auto hostData = toHost<float, 2>(
                (float*)x,
                resources_->getDefaultStream(config_.device),
                {n, this->d});
        trainQuantizer_(n, hostData.data());

        setIndex_(
                resources_.get(),
                this->d,
                this->nlist,
                this->metric_type,
                this->metric_arg,
                false,   // no residual
                nullptr, // no scalar quantizer
                ivfFlatConfig_.interleavedLayout,
                ivfFlatConfig_.indicesOptions,
                config_.memorySpace);
        updateQuantizer();
    }

    // The quantizer is now trained; construct the IVF index
    baseIndex_ = std::static_pointer_cast<IVFBase, IVFFlat>(index_);

    if (reserveMemoryVecs_) {
        if (should_use_cuvs(config_)) {
            FAISS_THROW_MSG(
                    "Pre-allocation of IVF lists is not supported with cuVS enabled.");
        } else
            index_->reserveMemory(reserveMemoryVecs_);
    }

    this->is_trained = true;
}

void GpuIndexIVFFlat::setIndex_(
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
        MemorySpace space) {
    if (should_use_cuvs(config_)) {
#if defined USE_NVIDIA_CUVS
        FAISS_THROW_IF_NOT_MSG(
                ivfFlatConfig_.indicesOptions == INDICES_64_BIT,
                "cuVS only supports INDICES_64_BIT");
        if (!ivfFlatConfig_.interleavedLayout) {
            fprintf(stderr,
                    "WARN: interleavedLayout is set to False with cuVS enabled. This will be ignored.\n");
        }
        index_.reset(new CuvsIVFFlat(
                resources,
                dim,
                nlist,
                metric,
                metricArg,
                useResidual,
                scalarQ,
                interleavedLayout,
                indicesOptions,
                space));
#else
        FAISS_THROW_MSG(
                "cuVS has not been compiled into the current version so it cannot be used.");
#endif
    } else {
        index_.reset(new IVFFlat(
                resources,
                dim,
                nlist,
                metric,
                metricArg,
                useResidual,
                scalarQ,
                interleavedLayout,
                indicesOptions,
                space));
    }
}

void GpuIndexIVFFlat::reconstruct_n(idx_t i0, idx_t ni, float* out) const {
    FAISS_ASSERT(index_);

    if (ni == 0) {
        // nothing to do
        return;
    }

    FAISS_THROW_IF_NOT_FMT(
            i0 < this->ntotal,
            "start index (%zu) out of bounds (ntotal %zu)",
            i0,
            this->ntotal);
    FAISS_THROW_IF_NOT_FMT(
            i0 + ni - 1 < this->ntotal,
            "max index requested (%zu) out of bounds (ntotal %zu)",
            i0 + ni - 1,
            this->ntotal);

    index_->reconstruct_n(i0, ni, out);
}

} // namespace gpu
} // namespace faiss
