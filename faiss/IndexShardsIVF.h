/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/IndexIVF.h>
#include <faiss/IndexShards.h>
#include <faiss/IVFEvictLoadInterface.h>
#include <cstdint>
#include <vector>

namespace faiss {

/**
 * IndexShards with a common coarse quantizer. All the indexes added should be
 * IndexIVFInterface indexes so that the search_precomputed can be called.
 */
struct IndexShardsIVF : public IndexShards, Level1Quantizer {
    explicit IndexShardsIVF(
            Index* quantizer,
            size_t nlist,
            bool threaded = false,
            bool successive_ids = true);

    void addIndex(Index* index) override;

    void add_with_ids(idx_t n, const component_t* x, const idx_t* xids)
            override;

    void train(idx_t n, const component_t* x) override;

    void search(
            idx_t n,
            const component_t* x,
            idx_t k,
            distance_t* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    /// Set list-to-shard mapping for evict/load forwarding (used by multi-GPU
    /// IVF sharding). Must be called after add_shard. list_to_shard[listId]
    /// gives the shard index owning that list.
    void setListToShardMapping(const std::vector<int>& list_to_shard);

    /// Evict/load forwarding: dispatch to the shard that owns each list.
    /// Requires setListToShardMapping to have been called and shards to be
    /// IVFEvictLoadInterface (e.g. GpuIndexIVFFlat).
    size_t evictCentroidToCpu(idx_t listId);
    size_t loadCentroidToGpu(idx_t listId);
    std::vector<uint64_t> evictCentroidsToCpu(
            const std::vector<idx_t>& listIds);
    std::vector<uint64_t> loadCentroidsToGpu(
            const std::vector<idx_t>& listIds);
    bool isListOnGpu(idx_t listId) const;
    std::vector<idx_t> getEvictedLists() const;

 private:
    std::vector<int> list_to_shard_;
};

} // namespace faiss
