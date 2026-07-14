#include "retrieval_engine.hpp"
#include "hnsw_index.hpp"
#include "memory_pool.hpp"
#include "simd_dot.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <new>
#include <algorithm>
#include <random>
#include <atomic>

namespace retrieval {
namespace hnsw {

struct RetrievalEngineImpl {
    HNSWIndex index;
    retrieval::memory::ArenaPool pool;
    retrieval::memory::ArenaPool temp_pool;
    BuildConfig config;
    bool built = false;
};

static inline RetrievalError to_error(int code) {
    return static_cast<RetrievalError>(code);
}

static inline RetrievalEngineImpl* get_impl(RetrievalEngine* engine) {
    return reinterpret_cast<RetrievalEngineImpl*>(engine);
}

static inline const RetrievalEngineImpl* get_impl(const RetrievalEngine* engine) {
    return reinterpret_cast<const RetrievalEngineImpl*>(engine);
}

}  // namespace hnsw
}  // namespace retrieval

extern "C" {

RetrievalError retrieval_engine_create(RetrievalEngine** engine, const RetrievalConfig* config) {
    if (!engine) return RETRIEVAL_ERROR_INVALID_ARGUMENT;

    retrieval::hnsw::RetrievalEngineImpl* impl = new (std::nothrow) retrieval::hnsw::RetrievalEngineImpl();
    if (!impl) return RETRIEVAL_ERROR_OUT_OF_MEMORY;

    if (config) {
        impl->config.max_elements = config->max_elements > 0 ? config->max_elements : retrieval::hnsw::MAX_ELEMENTS;
        impl->config.M = config->M > 0 ? config->M : 32;
        impl->config.ef_construction = config->ef_construction > 0 ? config->ef_construction : 200;
        impl->config.ef_search = config->ef_search > 0 ? config->ef_search : 100;
        impl->config.random_seed = config->random_seed;
        impl->config.pool_size = config->pool_size_bytes > 0 ? config->pool_size_bytes : (2ull * 1024 * 1024 * 1024);
    } else {
        impl->config.max_elements = retrieval::hnsw::MAX_ELEMENTS;
        impl->config.M = 32;
        impl->config.ef_construction = 200;
        impl->config.ef_search = 100;
        impl->config.random_seed = 42;
        impl->config.pool_size = 2ull * 1024 * 1024 * 1024;
    }

    impl->index.max_elements = impl->config.max_elements;
    impl->index.max_layers = retrieval::hnsw::MAX_LAYERS;
    impl->index.max_m = impl->config.M;
    impl->index.max_m0 = impl->config.M * 2;
    impl->index.ef_construction = impl->config.ef_construction;
    impl->index.ef_search = impl->config.ef_search;
    impl->index.current_count = 0;
    impl->index.entry_point = UINT32_MAX;
    impl->index.allocator = &impl->pool;
    impl->index.insert_count = 0;
    impl->index.search_count = 0;

    *engine = reinterpret_cast<RetrievalEngine*>(impl);
    return RETRIEVAL_SUCCESS;
}

void retrieval_engine_destroy(RetrievalEngine* engine) {
    if (!engine) return;
    retrieval::hnsw::RetrievalEngineImpl* impl = retrieval::hnsw::get_impl(engine);
    delete impl;
}

RetrievalError retrieval_engine_build(RetrievalEngine* engine, const float* vectors, uint32_t count, uint32_t dim) {
    if (!engine || !vectors) return RETRIEVAL_ERROR_INVALID_ARGUMENT;
    if (count == 0 || dim != retrieval::hnsw::EMBEDDING_DIM) return RETRIEVAL_ERROR_INVALID_ARGUMENT;

    retrieval::hnsw::RetrievalEngineImpl* impl = retrieval::hnsw::get_impl(engine);

    if (count > impl->config.max_elements) {
        return RETRIEVAL_ERROR_INVALID_ARGUMENT;
    }

    impl->index.nodes = static_cast<retrieval::hnsw::Node*>(impl->pool.allocate(
        impl->config.max_elements * sizeof(retrieval::hnsw::Node), alignof(retrieval::hnsw::Node)
    ));
    if (!impl->index.nodes) {
        return RETRIEVAL_ERROR_OUT_OF_MEMORY;
    }

    std::mt19937 rng(impl->config.random_seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (uint32_t i = 0; i < count; ++i) {
        retrieval::hnsw::Node* node = &impl->index.nodes[i];
        node->id = i;
        node->layer = retrieval::hnsw::get_random_layer(
            retrieval::hnsw::MULT, retrieval::hnsw::MAX_LAYERS,
            static_cast<uint32_t>(dist(rng) * 4294967295.0f)
        );
        for (uint32_t l = 0; l < retrieval::hnsw::MAX_LAYERS; ++l) {
            node->neighbor_count[l] = 0;
        }
        std::memcpy(node->vector, vectors + i * dim, dim * sizeof(float));
    }

    impl->index.current_count = count;

    if (count > 0) {
        impl->index.entry_point = 0;
        for (uint32_t i = 1; i < count; ++i) {
            if (impl->index.nodes[i].layer > impl->index.nodes[impl->index.entry_point].layer) {
                impl->index.entry_point = i;
            }
        }
    }

    for (uint32_t i = 0; i < count; ++i) {
        if (i % 1000 == 0 && impl->temp_pool.exhausted()) {
            impl->temp_pool.reset();
        }

        uint32_t ep = impl->index.entry_point;
        uint32_t layer = impl->index.nodes[i].layer;

        for (int l = static_cast<int>(retrieval::hnsw::MAX_LAYERS) - 1; l > static_cast<int>(layer); --l) {
            uint32_t ef = 1;
            retrieval::memory::ThreadLocalArena temp_arena(&impl->temp_pool);
            uint32_t* candidates = static_cast<uint32_t*>(temp_arena.allocate(impl->config.max_elements * sizeof(uint32_t)));
            float* cand_dists = static_cast<float*>(temp_arena.allocate(impl->config.max_elements * sizeof(float)));
            uint32_t cand_count = 0;

            retrieval::hnsw::search_layer(&impl->index, impl->index.nodes[i].vector, l, ef, ep, &impl->temp_pool, candidates, cand_dists, &cand_count);
            ep = candidates[0];
        }

        for (int l = static_cast<int>(layer); l >= 0; --l) {
            uint32_t ef = (l == static_cast<int>(layer)) ? impl->config.ef_construction : 1;
            retrieval::memory::ThreadLocalArena temp_arena(&impl->temp_pool);
            uint32_t* candidates = static_cast<uint32_t*>(temp_arena.allocate(impl->config.max_elements * sizeof(uint32_t)));
            float* cand_dists = static_cast<float*>(temp_arena.allocate(impl->config.max_elements * sizeof(float)));
            uint32_t cand_count = 0;

            retrieval::hnsw::search_layer(&impl->index, impl->index.nodes[i].vector, l, ef, ep, &impl->temp_pool, candidates, cand_dists, &cand_count);

            uint32_t selected[64];
            uint32_t selected_count = 0;
            retrieval::hnsw::heuristic_select_neighbors(cand_dists, candidates, cand_count, impl->config.M, selected, &selected_count);

            for (uint32_t j = 0; j < selected_count; ++j) {
                uint32_t neighbor_id = selected[j];
                retrieval::hnsw::Node* neighbor = &impl->index.nodes[neighbor_id];
                retrieval::hnsw::mutual_link(&impl->index.nodes[i], neighbor, l, impl->config.M, &impl->pool);
            }

            for (uint32_t j = 0; j < selected_count; ++j) {
                uint32_t neighbor_id = selected[j];
                retrieval::hnsw::Node* neighbor = &impl->index.nodes[neighbor_id];
                if (neighbor->neighbor_count[l] > impl->config.M) {
                    retrieval::hnsw::unlink(neighbor, l, i, &impl->pool);
                }
            }

            if (l == 0) {
                ep = candidates[0];
            }
        }

        impl->index.insert_count.fetch_add(1, std::memory_order_relaxed);
    }

    impl->built = true;
    return RETRIEVAL_SUCCESS;
}

RetrievalError retrieval_engine_search(
    RetrievalEngine* engine,
    const float* query,
    uint32_t dim,
    uint32_t k,
    uint32_t ef_search,
    uint32_t* out_indices,
    float* out_distances
) {
    if (!engine || !query || !out_indices || !out_distances) return RETRIEVAL_ERROR_INVALID_ARGUMENT;
    if (dim != retrieval::hnsw::EMBEDDING_DIM || k == 0) return RETRIEVAL_ERROR_INVALID_ARGUMENT;

    retrieval::hnsw::RetrievalEngineImpl* impl = retrieval::hnsw::get_impl(engine);
    if (!impl->built) return RETRIEVAL_ERROR_INDEX_NOT_BUILT;

    uint32_t ef = ef_search > 0 ? ef_search : impl->config.ef_search;
    retrieval::memory::ThreadLocalArena temp_arena(&impl->temp_pool);

    uint32_t* candidates = static_cast<uint32_t*>(temp_arena.allocate(impl->config.max_elements * sizeof(uint32_t)));
    float* cand_dists = static_cast<float*>(temp_arena.allocate(impl->config.max_elements * sizeof(float)));
    uint32_t cand_count = 0;

    uint32_t ep = impl->index.entry_point;
    for (int l = retrieval::hnsw::MAX_LAYERS - 1; l >= 0; --l) {
        retrieval::hnsw::search_layer(&impl->index, query, l, ef, ep, &impl->temp_pool, candidates, cand_dists, &cand_count);
        if (cand_count > 0) ep = candidates[0];
    }

    impl->index.search_count.fetch_add(1, std::memory_order_relaxed);

    std::partial_sort_copy(candidates, candidates + cand_count,
                          out_indices, out_indices + k,
                          [&](uint32_t a, uint32_t b) {
                              float da = retrieval::hnsw::distance_l2sq(query, impl->index.nodes[a].vector, dim);
                              float db = retrieval::hnsw::distance_l2sq(query, impl->index.nodes[b].vector, dim);
                              return da < db;
                          });

    for (uint32_t i = 0; i < k; ++i) {
        out_distances[i] = retrieval::hnsw::distance_l2sq(query, impl->index.nodes[out_indices[i]].vector, dim);
    }

    return RETRIEVAL_SUCCESS;
}

RetrievalError retrieval_engine_batch_search(
    RetrievalEngine* engine,
    const float* queries,
    uint32_t num_queries,
    uint32_t dim,
    uint32_t k,
    uint32_t ef_search,
    uint32_t* out_indices,
    float* out_distances
) {
    if (!engine || !queries || !out_indices || !out_distances) return RETRIEVAL_ERROR_INVALID_ARGUMENT;
    if (num_queries == 0 || dim != retrieval::hnsw::EMBEDDING_DIM || k == 0) return RETRIEVAL_ERROR_INVALID_ARGUMENT;

    for (uint32_t i = 0; i < num_queries; ++i) {
        RetrievalError err = retrieval_engine_search(engine, queries + i * dim, dim, k, ef_search,
                                                      out_indices + i * k, out_distances + i * k);
        if (err != RETRIEVAL_SUCCESS) return err;
    }
    return RETRIEVAL_SUCCESS;
}

RetrievalError retrieval_engine_save(const RetrievalEngine* engine, const char* path) {
    if (!engine || !path) return RETRIEVAL_ERROR_INVALID_ARGUMENT;
    const retrieval::hnsw::RetrievalEngineImpl* impl = retrieval::hnsw::get_impl(engine);
    if (!impl->built) return RETRIEVAL_ERROR_INDEX_NOT_BUILT;

    FILE* f = std::fopen(path, "wb");
    if (!f) return RETRIEVAL_ERROR_FILE_IO;

    uint32_t magic = 0x484E5357;
    std::fwrite(&magic, sizeof(magic), 1, f);
    std::fwrite(&impl->config, sizeof(impl->config), 1, f);
    std::fwrite(&impl->index.current_count, sizeof(uint32_t), 1, f);
    std::fwrite(&impl->index.entry_point, sizeof(uint32_t), 1, f);
    std::fwrite(&impl->index.ef_construction, sizeof(uint32_t), 1, f);
    std::fwrite(&impl->index.ef_search, sizeof(uint32_t), 1, f);

    for (uint32_t i = 0; i < impl->index.current_count; ++i) {
        const retrieval::hnsw::Node* node = &impl->index.nodes[i];
        std::fwrite(&node->id, sizeof(uint32_t), 1, f);
        std::fwrite(&node->layer, sizeof(uint32_t), 1, f);
        for (uint32_t l = 0; l < retrieval::hnsw::MAX_LAYERS; ++l) {
            std::fwrite(&node->neighbor_count[l], sizeof(uint32_t), 1, f);
            std::fwrite(node->neighbors[l], sizeof(uint32_t), node->neighbor_count[l], f);
        }
        std::fwrite(node->vector, sizeof(float), retrieval::hnsw::EMBEDDING_DIM, f);
    }

    std::fclose(f);
    return RETRIEVAL_SUCCESS;
}

RetrievalError retrieval_engine_load(RetrievalEngine** engine, const char* path, const RetrievalConfig* config) {
    if (!engine || !path) return RETRIEVAL_ERROR_INVALID_ARGUMENT;

    FILE* f = std::fopen(path, "rb");
    if (!f) return RETRIEVAL_ERROR_FILE_IO;

    uint32_t magic;
    std::fread(&magic, sizeof(magic), 1, f);
    if (magic != 0x484E5357) {
        std::fclose(f);
        return RETRIEVAL_ERROR_INVALID_ARGUMENT;
    }

    RetrievalError err = retrieval_engine_create(engine, config);
    if (err != RETRIEVAL_SUCCESS) {
        std::fclose(f);
        return err;
    }

    retrieval::hnsw::RetrievalEngineImpl* impl = retrieval::hnsw::get_impl(*engine);
    std::fread(&impl->config, sizeof(impl->config), 1, f);
    std::fread(&impl->index.current_count, sizeof(uint32_t), 1, f);
    std::fread(&impl->index.entry_point, sizeof(uint32_t), 1, f);
    std::fread(&impl->index.ef_construction, sizeof(uint32_t), 1, f);
    std::fread(&impl->index.ef_search, sizeof(uint32_t), 1, f);

    impl->index.nodes = static_cast<retrieval::hnsw::Node*>(impl->pool.allocate(
        impl->config.max_elements * sizeof(retrieval::hnsw::Node), alignof(retrieval::hnsw::Node)
    ));
    if (!impl->index.nodes) {
        std::fclose(f);
        retrieval_engine_destroy(*engine);
        return RETRIEVAL_ERROR_OUT_OF_MEMORY;
    }

    for (uint32_t i = 0; i < impl->index.current_count; ++i) {
        retrieval::hnsw::Node* node = &impl->index.nodes[i];
        std::fread(&node->id, sizeof(uint32_t), 1, f);
        std::fread(&node->layer, sizeof(uint32_t), 1, f);
        for (uint32_t l = 0; l < retrieval::hnsw::MAX_LAYERS; ++l) {
            std::fread(&node->neighbor_count[l], sizeof(uint32_t), 1, f);
            std::fread(node->neighbors[l], sizeof(uint32_t), node->neighbor_count[l], f);
        }
        std::fread(node->vector, sizeof(float), retrieval::hnsw::EMBEDDING_DIM, f);
    }

    std::fclose(f);
    impl->built = true;
    return RETRIEVAL_SUCCESS;
}

RetrievalError retrieval_engine_get_stats(
    const RetrievalEngine* engine,
    uint32_t* element_count,
    uint32_t* max_elements,
    size_t* memory_used_bytes,
    uint64_t* insert_count,
    uint64_t* search_count
) {
    if (!engine) return RETRIEVAL_ERROR_INVALID_ARGUMENT;
    const retrieval::hnsw::RetrievalEngineImpl* impl = retrieval::hnsw::get_impl(engine);

    if (element_count) *element_count = impl->index.current_count;
    if (max_elements) *max_elements = impl->config.max_elements;
    if (memory_used_bytes) *memory_used_bytes = impl->pool.total_allocated();
    if (insert_count) *insert_count = impl->index.insert_count.load(std::memory_order_relaxed);
    if (search_count) *search_count = impl->index.search_count.load(std::memory_order_relaxed);
    return RETRIEVAL_SUCCESS;
}

RetrievalError retrieval_engine_set_ef_search(RetrievalEngine* engine, uint32_t ef_search) {
    if (!engine) return RETRIEVAL_ERROR_INVALID_ARGUMENT;
    retrieval::hnsw::RetrievalEngineImpl* impl = retrieval::hnsw::get_impl(engine);
    impl->config.ef_search = ef_search;
    impl->index.ef_search = ef_search;
    return RETRIEVAL_SUCCESS;
}

RetrievalError retrieval_engine_get_ef_search(const RetrievalEngine* engine, uint32_t* ef_search) {
    if (!engine || !ef_search) return RETRIEVAL_ERROR_INVALID_ARGUMENT;
    const retrieval::hnsw::RetrievalEngineImpl* impl = retrieval::hnsw::get_impl(engine);
    *ef_search = impl->config.ef_search;
    return RETRIEVAL_SUCCESS;
}

RetrievalError retrieval_engine_prefetch(const RetrievalEngine* engine, const float* queries, uint32_t num_queries, uint32_t dim) {
    if (!engine || !queries || dim != retrieval::hnsw::EMBEDDING_DIM) return RETRIEVAL_ERROR_INVALID_ARGUMENT;
    const retrieval::hnsw::RetrievalEngineImpl* impl = retrieval::hnsw::get_impl(engine);
    if (!impl->built) return RETRIEVAL_ERROR_INDEX_NOT_BUILT;

    for (uint32_t i = 0; i < num_queries; ++i) {
        __builtin_prefetch(queries + i * dim);
    }
    return RETRIEVAL_SUCCESS;
}

const char* retrieval_error_string(RetrievalError error) {
    switch (error) {
        case RETRIEVAL_SUCCESS: return "Success";
        case RETRIEVAL_ERROR_INVALID_ARGUMENT: return "Invalid argument";
        case RETRIEVAL_ERROR_OUT_OF_MEMORY: return "Out of memory";
        case RETRIEVAL_ERROR_INDEX_NOT_BUILT: return "Index not built";
        case RETRIEVAL_ERROR_FILE_IO: return "I/O error";
        default: return "Unknown error";
    }
}

}

namespace retrieval {

RetrievalEngineWrapper::RetrievalEngineWrapper(const RetrievalConfig& config)
    : engine_(nullptr), config_(config), built_(false) {
    RetrievalError err = retrieval_engine_create(&engine_, &config_);
    if (err != RETRIEVAL_SUCCESS) {
        throw std::runtime_error(retrieval_error_string(err));
    }
}

RetrievalEngineWrapper::~RetrievalEngineWrapper() {
    if (engine_) {
        retrieval_engine_destroy(engine_);
    }
}

RetrievalEngineWrapper::RetrievalEngineWrapper(RetrievalEngineWrapper&& other) noexcept
    : engine_(other.engine_), config_(other.config_), built_(other.built_) {
    other.engine_ = nullptr;
    other.built_ = false;
}

RetrievalEngineWrapper& RetrievalEngineWrapper::operator=(RetrievalEngineWrapper&& other) noexcept {
    if (this != &other) {
        if (engine_) retrieval_engine_destroy(engine_);
        engine_ = other.engine_;
        config_ = other.config_;
        built_ = other.built_;
        other.engine_ = nullptr;
        other.built_ = false;
    }
    return *this;
}

RetrievalError RetrievalEngineWrapper::build(const float* vectors, uint32_t count, uint32_t dim) {
    RetrievalError err = retrieval_engine_build(engine_, vectors, count, dim);
    if (err == RETRIEVAL_SUCCESS) built_ = true;
    return err;
}

RetrievalError RetrievalEngineWrapper::search(const float* query, uint32_t dim, uint32_t k, uint32_t ef_search,
                                               std::vector<uint32_t>& out_indices, std::vector<float>& out_distances) const {
    out_indices.resize(k);
    out_distances.resize(k);
    return retrieval_engine_search(engine_, query, dim, k, ef_search, out_indices.data(), out_distances.data());
}

RetrievalError RetrievalEngineWrapper::batch_search(const float* queries, uint32_t num_queries, uint32_t dim, uint32_t k, uint32_t ef_search,
                                                     std::vector<uint32_t>& out_indices, std::vector<float>& out_distances) const {
    out_indices.resize(num_queries * k);
    out_distances.resize(num_queries * k);
    return retrieval_engine_batch_search(engine_, queries, num_queries, dim, k, ef_search, out_indices.data(), out_distances.data());
}

RetrievalError RetrievalEngineWrapper::save(const std::string& path) const {
    return retrieval_engine_save(engine_, path.c_str());
}

RetrievalError RetrievalEngineWrapper::load(const std::string& path, const RetrievalConfig& config) {
    RetrievalEngine* new_engine = nullptr;
    RetrievalError err = retrieval_engine_load(&new_engine, path.c_str(), &config);
    if (err == RETRIEVAL_SUCCESS) {
        if (engine_) retrieval_engine_destroy(engine_);
        engine_ = new_engine;
        config_ = config;
        built_ = true;
    }
    return err;
}

void RetrievalEngineWrapper::get_stats(uint32_t& element_count, uint32_t& max_elements, size_t& memory_used,
                                        uint64_t& insert_count, uint64_t& search_count) const {
    retrieval_engine_get_stats(engine_, &element_count, &max_elements, &memory_used, &insert_count, &search_count);
}

void RetrievalEngineWrapper::set_ef_search(uint32_t ef_search) {
    retrieval_engine_set_ef_search(engine_, ef_search);
}

uint32_t RetrievalEngineWrapper::get_ef_search() const {
    uint32_t ef = 0;
    retrieval_engine_get_ef_search(engine_, &ef);
    return ef;
}

}  // namespace retrieval