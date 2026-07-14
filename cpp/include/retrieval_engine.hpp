#pragma once

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RetrievalEngine RetrievalEngine;

typedef enum {
    RETRIEVAL_SUCCESS = 0,
    RETRIEVAL_ERROR_NULL_POINTER = -1,
    RETRIEVAL_ERROR_INVALID_ARGUMENT = -2,
    RETRIEVAL_ERROR_OUT_OF_MEMORY = -3,
    RETRIEVAL_ERROR_INDEX_NOT_BUILT = -4,
    RETRIEVAL_ERROR_DIMENSION_MISMATCH = -5,
    RETRIEVAL_ERROR_FILE_IO = -6,
    RETRIEVAL_ERROR_ALREADY_INITIALIZED = -7
} RetrievalError;

typedef struct {
    uint32_t max_elements;
    uint32_t M;
    uint32_t ef_construction;
    uint32_t ef_search;
    uint32_t random_seed;
    size_t pool_size_bytes;
} RetrievalConfig;

static const RetrievalConfig RETRIEVAL_DEFAULT_CONFIG = {
    .max_elements = 70000,
    .M = 32,
    .ef_construction = 200,
    .ef_search = 500,  // must be >= default k (500) to avoid duplicate zero-index padding
    .random_seed = 42,
    .pool_size_bytes = 2ull * 1024 * 1024 * 1024
};

RetrievalError retrieval_engine_create(RetrievalEngine** engine, const RetrievalConfig* config);
void retrieval_engine_destroy(RetrievalEngine* engine);

RetrievalError retrieval_engine_build(RetrievalEngine* engine, const float* vectors, uint32_t count, uint32_t dim);

RetrievalError retrieval_engine_search(
    RetrievalEngine* engine,
    const float* query,
    uint32_t dim,
    uint32_t k,
    uint32_t ef_search,
    uint32_t* out_indices,
    float* out_distances
);

RetrievalError retrieval_engine_save(const RetrievalEngine* engine, const char* path);
RetrievalError retrieval_engine_load(RetrievalEngine** engine, const char* path, const RetrievalConfig* config);

RetrievalError retrieval_engine_get_stats(
    const RetrievalEngine* engine,
    uint32_t* element_count,
    uint32_t* max_elements,
    size_t* memory_used_bytes,
    uint64_t* insert_count,
    uint64_t* search_count
);

RetrievalError retrieval_engine_set_ef_search(RetrievalEngine* engine, uint32_t ef_search);
RetrievalError retrieval_engine_get_ef_search(const RetrievalEngine* engine, uint32_t* ef_search);

RetrievalError retrieval_engine_batch_search(
    RetrievalEngine* engine,
    const float* queries,
    uint32_t num_queries,
    uint32_t dim,
    uint32_t k,
    uint32_t ef_search,
    uint32_t* out_indices,
    float* out_distances
);

RetrievalError retrieval_engine_prefetch(const RetrievalEngine* engine, const float* queries, uint32_t num_queries, uint32_t dim);

const char* retrieval_error_string(RetrievalError error);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <vector>
#include <string>
#include <memory>

namespace retrieval {

class RetrievalEngineWrapper {
public:
    explicit RetrievalEngineWrapper(const RetrievalConfig& config = RETRIEVAL_DEFAULT_CONFIG);
    ~RetrievalEngineWrapper();

    RetrievalEngineWrapper(const RetrievalEngineWrapper&) = delete;
    RetrievalEngineWrapper& operator=(const RetrievalEngineWrapper&) = delete;

    RetrievalEngineWrapper(RetrievalEngineWrapper&& other) noexcept;
    RetrievalEngineWrapper& operator=(RetrievalEngineWrapper&& other) noexcept;

    RetrievalError build(const float* vectors, uint32_t count, uint32_t dim);
    RetrievalError search(const float* query, uint32_t dim, uint32_t k, uint32_t ef_search,
                          std::vector<uint32_t>& out_indices, std::vector<float>& out_distances) const;
    RetrievalError batch_search(const float* queries, uint32_t num_queries, uint32_t dim, uint32_t k, uint32_t ef_search,
                                std::vector<uint32_t>& out_indices, std::vector<float>& out_distances) const;

    RetrievalError save(const std::string& path) const;
    RetrievalError load(const std::string& path, const RetrievalConfig& config);

    void get_stats(uint32_t& element_count, uint32_t& max_elements, size_t& memory_used,
                   uint64_t& insert_count, uint64_t& search_count) const;

    void set_ef_search(uint32_t ef_search);
    uint32_t get_ef_search() const;

    bool is_built() const { return built_; }

private:
    RetrievalEngine* engine_;
    RetrievalConfig config_;
    bool built_;
};

}  // namespace retrieval
#endif