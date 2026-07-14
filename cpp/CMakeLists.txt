cmake_minimum_required(VERSION 3.16)
project(retrieval_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG")

if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64")
    set(CMAKE_MACOSX_RPATH ON)
endif()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1
)
FetchContent_MakeAvailable(pybind11)

add_library(retrieval_engine STATIC src/retrieval_engine.cpp)

target_include_directories(retrieval_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(retrieval_engine PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-parameter -Wno-sign-compare -Wno-c99-extensions)

pybind11_add_module(_cpp src/pybind_module.cpp)
target_link_libraries(_cpp PRIVATE retrieval_engine)

target_include_directories(_cpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${Python3_INCLUDE_DIRS})
set_target_properties(_cpp PROPERTIES PREFIX "" SUFFIX "${PYTHON_MODULE_EXTENSION}")
