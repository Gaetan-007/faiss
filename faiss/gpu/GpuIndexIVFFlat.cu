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

#if defined USE_NVIDIA_CUVS
#include <cuvs/neighbors/ivf_flat.hpp>
#include <faiss/gpu/utils/CuvsUtils.h>
#include <faiss/gpu/impl/CuvsIVFFlat.cuh>
#endif

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
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

    if (cpuListCache_.count(listId)) {
        return 0;
    }

    CpuListCache cache;
    cache.codes = index_->getListVectorData(listId, false);
    cache.ids = index_->getListIndices(listId);

    cpuListCache_.emplace(listId, std::move(cache));

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
    auto numVecs = (idx_t)cache.ids.size();

    if (ivfFlatConfig_.indicesOptions != INDICES_IVF) {
        FAISS_THROW_IF_NOT_MSG(
                !cache.ids.empty(),
                "Cached IVF list is missing indices for GPU load");
    }

    index_->addEncodedVectorsToListFromCpu(
            listId,
            cache.codes.data(),
            cache.ids.empty() ? nullptr : cache.ids.data(),
            numVecs);

    size_t bytesLoaded = cache.codes.size() + cache.ids.size() * sizeof(idx_t);
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
}

bool GpuIndexIVFFlat::isAutoFetchEnabled() const {
    return autoFetchEnabled_;
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

    // If auto-fetch is not enabled, just call parent implementation
    if (!autoFetchEnabled_ || cpuListCache_.empty()) {
        GpuIndexIVF::searchImpl_(n, x, k, distances, labels, params);
        return;
    }

    // We need to determine which IVF lists will be accessed
    // This requires a coarse quantizer search first
    int use_nprobe = getCurrentNProbe_(params);
    auto stream = resources_->getDefaultStream(config_.device);

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
    std::vector<idx_t> listIdsToCheck(uniqueListIds.begin(), uniqueListIds.end());

    // Fetch any missing lists from CPU cache
    // Note: we cast away const here because auto-fetch modifies internal state
    // This is acceptable because the logical result of the search doesn't change
    const_cast<GpuIndexIVFFlat*>(this)->fetchMissingListsForSearch_(
            listIdsToCheck);

    // Now perform the actual search
    GpuIndexIVF::searchImpl_(n, x, k, distances, labels, params);
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
