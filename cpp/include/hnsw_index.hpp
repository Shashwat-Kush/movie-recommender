#pragma once
#include <cstdint>
#include <cstddef>
#include <atomic>
#include <cstring>
#include <new>
#include <cmath>
#include <limits>
#include <algorithm>

#include "memory_pool.hpp"
#include "simd_dot.hpp"

namespace retrieval {
namespace hnsw {

static constexpr uint32_t MAX_LAYERS = 16;
static constexpr uint32_t MAX_M = 64;
static constexpr uint32_t MAX_M0 = 64;
static constexpr uint32_t MAX_ELEMENTS = 70000;
static constexpr uint32_t EMBEDDING_DIM = 128;
static const float MULT = 1.44269504089f;

struct alignas(64) Node {
    uint32_t id;
    uint32_t layer;
    uint32_t neighbor_count[MAX_LAYERS];
    uint32_t neighbors[MAX_LAYERS][MAX_M];
    float vector[EMBEDDING_DIM];
};

struct HNSWIndex {
    Node* nodes;
    uint32_t max_elements;
    uint32_t max_layers;
    uint32_t max_m;
    uint32_t max_m0;
    uint32_t ef_construction;
    uint32_t ef_search;
    uint32_t current_count;
    uint32_t entry_point;
    retrieval::ArenaPool* allocator;

    mutable std::atomic<uint32_t> insert_count;
    mutable std::atomic<uint32_t> search_count;
};

struct BuildConfig {
    uint32_t max_elements = MAX_ELEMENTS;
    uint32_t M = 32;
    uint32_t ef_construction = 200;
    uint32_t ef_search = 100;
    uint32_t random_seed = 42;
    size_t pool_size = 2ull * 1024 * 1024 * 1024;
};

inline uint32_t get_random_layer(float mult, uint32_t max_layers, uint32_t seed) noexcept {
    double r = (static_cast<double>(seed * 1664525 + 1013904223) / 4294967296.0);
    double level = -std::log(r) * mult;
    uint32_t layer = static_cast<uint32_t>(level);
    return layer < max_layers ? layer : max_layers - 1;
}

inline void heuristic_select_neighbors(
    const float* distances,
    const uint32_t* indices,
    uint32_t count,
    uint32_t M,
    uint32_t* selected,
    uint32_t* selected_count
) noexcept {
    if (count <= M) {
        for (uint32_t i = 0; i < count; ++i) selected[i] = indices[i];
        *selected_count = count;
        return;
    }
    uint32_t idx[2048];
    for (uint32_t i = 0; i < count; ++i) idx[i] = i;
    for (uint32_t i = 1; i < count; ++i) {
        uint32_t j = i;
        while (j > 0 && distances[idx[j - 1]] > distances[idx[j]]) {
            std::swap(idx[j], idx[j - 1]);
            --j;
        }
    }
    *selected_count = 0;
    for (uint32_t i = 0; i < count && *selected_count < M; ++i) {
        bool keep = true;
        for (uint32_t j = 0; j < *selected_count; ++j) {
            if (indices[idx[i]] == selected[j]) { keep = false; break; }
        }
        if (keep) {
            selected[*selected_count] = indices[idx[i]];
            (*selected_count)++;
        }
    }
}

inline void mutual_link(Node* node_a, Node* node_b, uint32_t layer, uint32_t M, retrieval::ArenaPool* alloc) noexcept {
    if (node_a->neighbor_count[layer] < M) {
        node_a->neighbors[layer][node_a->neighbor_count[layer]++] = node_b->id;
    }
    if (node_b->neighbor_count[layer] < M) {
        node_b->neighbors[layer][node_b->neighbor_count[layer]++] = node_a->id;
    }
}

inline void unlink(Node* node, uint32_t layer, uint32_t neighbor_id, retrieval::ArenaPool* alloc) noexcept {
    uint32_t count = node->neighbor_count[layer];
    for (uint32_t i = 0; i < count; ++i) {
        if (node->neighbors[layer][i] == neighbor_id) {
            node->neighbors[layer][i] = node->neighbors[layer][count - 1];
            node->neighbor_count[layer]--;
            break;
        }
    }
}

inline float distance_l2sq(const float* a, const float* b, uint32_t dim) noexcept {
    float dot = retrieval::simd::dot_product(a, b, dim);
    return -dot;  // higher dot = more similar for L2-normalized vectors; negate so SMALLER = better
}

inline void search_layer(
    const HNSWIndex* index, const float* query, uint32_t layer, uint32_t ef, uint32_t entry_point,
    retrieval::ArenaPool* temp_alloc, uint32_t* candidates, float* cand_dists, uint32_t* cand_count
) noexcept {
    const Node* entry_node = &index->nodes[entry_point];
    float entry_dist = distance_l2sq(query, entry_node->vector, EMBEDDING_DIM);

    *cand_count = 1;
    candidates[0] = entry_point;
    cand_dists[0] = entry_dist;

    uint32_t visited_idx = 0;
    bool* visited = static_cast<bool*>(temp_alloc->allocate(index->max_elements * sizeof(bool), alignof(bool)));
    std::memset(visited, 0, index->max_elements * sizeof(bool));
    visited[entry_point] = true;

    while (visited_idx < *cand_count && visited_idx < ef) {
        uint32_t curr_id = candidates[visited_idx];
        visited_idx++;

        const Node* curr_node = &index->nodes[curr_id];
        uint32_t neighbor_count = curr_node->neighbor_count[layer];

        for (uint32_t i = 0; i < neighbor_count; ++i) {
            uint32_t neighbor_id = curr_node->neighbors[layer][i];
            if (visited[neighbor_id]) continue;
            visited[neighbor_id] = true;

            float dist = distance_l2sq(query, index->nodes[neighbor_id].vector, EMBEDDING_DIM);

            if (*cand_count < ef) {
                candidates[*cand_count] = neighbor_id;
                cand_dists[*cand_count] = dist;
                (*cand_count)++;
            } else {
                uint32_t max_idx = 0;
                for (uint32_t j = 1; j < ef; ++j) {
                    if (cand_dists[j] > cand_dists[max_idx]) max_idx = j;
                }
                if (dist < cand_dists[max_idx]) {
                    candidates[max_idx] = neighbor_id;
                    cand_dists[max_idx] = dist;
                }
            }
        }
    }

    uint32_t idx[2048];
    for (uint32_t i = 0; i < *cand_count; ++i) idx[i] = i;
    for (uint32_t i = 1; i < *cand_count; ++i) {
        uint32_t j = i;
        while (j > 0 && cand_dists[idx[j - 1]] > cand_dists[idx[j]]) {
            std::swap(idx[j], idx[j - 1]);
            --j;
        }
    }

    uint32_t* sorted_candidates = static_cast<uint32_t*>(temp_alloc->allocate(ef * sizeof(uint32_t), alignof(uint32_t)));
    float* sorted_dists = static_cast<float*>(temp_alloc->allocate(ef * sizeof(float), alignof(float)));

    for (uint32_t i = 0; i < *cand_count; ++i) {
        sorted_candidates[i] = candidates[idx[i]];
        sorted_dists[i] = cand_dists[idx[i]];
    }

    std::memcpy(candidates, sorted_candidates, *cand_count * sizeof(uint32_t));
    std::memcpy(cand_dists, sorted_dists, *cand_count * sizeof(float));
}

}  // namespace hnsw
}  // namespace retrieval