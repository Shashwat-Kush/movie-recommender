#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <stdexcept>

#include "retrieval_engine.hpp"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace retrieval;

namespace {

py::array_t<uint32_t> make_indices_array(uint32_t* data, size_t size) {
    return py::array_t<uint32_t>(
        {size},
        {sizeof(uint32_t)},
        data,
        py::cast([](uint32_t*) {})  // no-op deleter, memory owned by engine
    );
}

py::array_t<float> make_distances_array(float* data, size_t size) {
    return py::array_t<float>(
        {size},
        {sizeof(float)},
        data,
        py::cast([](float*) {})
    );
}

py::array_t<uint32_t> make_indices_copy(const std::vector<uint32_t>& vec) {
    py::array_t<uint32_t> arr(vec.size());
    std::memcpy(arr.mutable_data(), vec.data(), vec.size() * sizeof(uint32_t));
    return arr;
}

py::array_t<float> make_distances_copy(const std::vector<float>& vec) {
    py::array_t<float> arr(vec.size());
    std::memcpy(arr.mutable_data(), vec.data(), vec.size() * sizeof(float));
    return arr;
}

}  // anonymous namespace

PYBIND11_MODULE(_cpp, m) {
    m.doc() = "Movie Recommender C++ Retrieval Engine (HNSW + SIMD)";

    py::enum_<RetrievalError>(m, "RetrievalError")
        .value("SUCCESS", RETRIEVAL_SUCCESS)
        .value("INVALID_ARGUMENT", RETRIEVAL_ERROR_INVALID_ARGUMENT)
        .value("OUT_OF_MEMORY", RETRIEVAL_ERROR_OUT_OF_MEMORY)
        .value("INDEX_NOT_BUILT", RETRIEVAL_ERROR_INDEX_NOT_BUILT)
        .value("DIMENSION_MISMATCH", RETRIEVAL_ERROR_DIMENSION_MISMATCH)
        .value("FILE_IO", RETRIEVAL_ERROR_FILE_IO)
        .value("ALREADY_INITIALIZED", RETRIEVAL_ERROR_ALREADY_INITIALIZED);

    py::class_<RetrievalConfig>(m, "RetrievalConfig")
        .def(py::init<>())
        .def_readwrite("max_elements", &RetrievalConfig::max_elements)
        .def_readwrite("M", &RetrievalConfig::M)
        .def_readwrite("ef_construction", &RetrievalConfig::ef_construction)
        .def_readwrite("ef_search", &RetrievalConfig::ef_search)
        .def_readwrite("random_seed", &RetrievalConfig::random_seed)
        .def_readwrite("pool_size_bytes", &RetrievalConfig::pool_size_bytes);

    py::class_<RetrievalEngineWrapper>(m, "RetrievalEngine")
        .def(py::init([](const RetrievalConfig& config) {
            return std::make_unique<RetrievalEngineWrapper>(config);
        }), py::arg("config") = RetrievalConfig{})
        .def("build", [](RetrievalEngineWrapper& self, py::array_t<float, py::array::c_style | py::array::forcecast> vectors) {
            py::buffer_info buf = vectors.request();
            if (buf.ndim != 2) {
                throw py::value_error("vectors must be 2D array (n_vectors, dim)");
            }
            uint32_t count = static_cast<uint32_t>(buf.shape[0]);
            uint32_t dim = static_cast<uint32_t>(buf.shape[1]);
            float* ptr = static_cast<float*>(buf.ptr);
            RetrievalError err = self.build(ptr, count, dim);
            if (err != RETRIEVAL_SUCCESS) {
                throw std::runtime_error("Build failed: " + std::string(retrieval_error_string(err)));
            }
        }, py::arg("vectors"), "Build HNSW index from float32 vectors (n, 128)")
        .def("search", [](RetrievalEngineWrapper& self, py::array_t<float, py::array::c_style | py::array::forcecast> query,
                          uint32_t k, uint32_t ef_search) {
            py::buffer_info buf = query.request();
            if (buf.ndim != 1) {
                throw py::value_error("query must be 1D array (dim,)");
            }
            uint32_t dim = static_cast<uint32_t>(buf.shape[0]);
            float* ptr = static_cast<float*>(buf.ptr);

            if (ef_search < k) ef_search = k;  // clamp to avoid duplicate zero-index padding

            std::vector<uint32_t> indices(k);
            std::vector<float> distances(k);
            RetrievalError err = self.search(ptr, dim, k, ef_search, indices, distances);
            if (err != RETRIEVAL_SUCCESS) {
                throw std::runtime_error("Search failed: " + std::string(retrieval_error_string(err)));
            }
            return py::make_tuple(make_indices_copy(indices), make_distances_copy(distances));
        }, py::arg("query"), py::arg("k") = 500, py::arg("ef_search") = 100,
           "Search for k nearest neighbors, returns (indices, distances)")
.def("batch_search", [](RetrievalEngineWrapper& self, py::array_t<float, py::array::c_style | py::array::forcecast> queries,
                                 uint32_t k, uint32_t ef_search) {
            py::buffer_info buf = queries.request();
            if (buf.ndim != 2) {
                throw py::value_error("queries must be 2D array (n_queries, dim)");
            }
            uint32_t num_queries = static_cast<uint32_t>(buf.shape[0]);
            uint32_t dim = static_cast<uint32_t>(buf.shape[1]);
            float* ptr = static_cast<float*>(buf.ptr);

            if (ef_search < k) ef_search = k;  // clamp

            std::vector<uint32_t> indices(num_queries * k);
            std::vector<float> distances(num_queries * k);
            RetrievalError err = self.batch_search(ptr, num_queries, dim, k, ef_search, indices, distances);
            if (err != RETRIEVAL_SUCCESS) {
                throw std::runtime_error("Batch search failed: " + std::string(retrieval_error_string(err)));
            }

            py::array_t<uint32_t> idx_arr({num_queries, k});
            py::array_t<float> dist_arr({num_queries, k});
            std::memcpy(idx_arr.mutable_data(), indices.data(), indices.size() * sizeof(uint32_t));
            std::memcpy(dist_arr.mutable_data(), distances.data(), distances.size() * sizeof(float));
            return py::make_tuple(idx_arr, dist_arr);
        }, py::arg("queries"), py::arg("k") = 500, py::arg("ef_search") = 100,
           "Batch search for multiple queries, returns (indices[n,q], distances[n,q])")
        .def("save", [](RetrievalEngineWrapper& self, const std::string& path) {
            RetrievalError err = self.save(path);
            if (err != RETRIEVAL_SUCCESS) {
                throw std::runtime_error("Save failed: " + std::string(retrieval_error_string(err)));
            }
        }, py::arg("path"), "Save index to disk")
        .def("load", [](RetrievalEngineWrapper& self, const std::string& path, const RetrievalConfig& config) {
            RetrievalError err = self.load(path, config);
            if (err != RETRIEVAL_SUCCESS) {
                throw std::runtime_error("Load failed: " + std::string(retrieval_error_string(err)));
            }
        }, py::arg("path"), py::arg("config"), "Load index from disk")
        .def("get_stats", [](RetrievalEngineWrapper& self) {
            uint32_t element_count, max_elements;
            size_t memory_used;
            uint64_t insert_count, search_count;
            self.get_stats(element_count, max_elements, memory_used, insert_count, search_count);
            return py::dict(
                "element_count"_a = element_count,
                "max_elements"_a = max_elements,
                "memory_used_bytes"_a = memory_used,
                "insert_count"_a = insert_count,
                "search_count"_a = search_count
            );
        }, "Get engine statistics")
        .def("set_ef_search", &RetrievalEngineWrapper::set_ef_search, py::arg("ef_search"))
        .def("get_ef_search", &RetrievalEngineWrapper::get_ef_search)
        .def("is_built", &RetrievalEngineWrapper::is_built);

    m.attr("RETRIEVAL_DEFAULT_CONFIG") = RetrievalConfig{};
}