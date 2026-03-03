/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/MetricType.h>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace faiss {

/** Interface for IVF indices that support evicting/loading centroid lists
 * to/from CPU. Used by IndexShardsIVF to forward evict/load calls to shards
 * (e.g. GpuIndexIVFFlat).
 */
struct IVFEvictLoadInterface {
    virtual ~IVFEvictLoadInterface() = default;

    virtual size_t evictCentroidToCpu(idx_t listId) = 0;
    virtual size_t loadCentroidToGpu(idx_t listId) = 0;
    virtual std::vector<uint64_t> evictCentroidsToCpu(
            const std::vector<idx_t>& listIds) = 0;
    virtual std::vector<uint64_t> loadCentroidsToGpu(
            const std::vector<idx_t>& listIds) = 0;
    virtual bool isListOnGpu(idx_t listId) const = 0;
    virtual std::vector<idx_t> getEvictedLists() const = 0;
};

} // namespace faiss
