#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build_arm64"

echo "Building retrieval_engine for Apple Silicon..."

rm -rf "${BUILD_DIR}"

# Completely annihilate Conda's path overrides
unset CXX CC CFLAGS CXXFLAGS LDFLAGS CPPFLAGS 
unset CPLUS_INCLUDE_PATH C_INCLUDE_PATH CPATH LIBRARY_PATH LD_LIBRARY_PATH

export SDKROOT="$(xcrun --show-sdk-path)"
export CXX="$(xcrun -find c++)"
export CC="$(xcrun -find cc)"

# Force the C++ standard library path explicitly
export CXXFLAGS="-isystem ${SDKROOT}/usr/include/c++/v1 -isysroot ${SDKROOT} -stdlib=libc++"

cmake -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0 \
    -DCMAKE_OSX_SYSROOT="${SDKROOT}" \
    "${SCRIPT_DIR}"

cmake --build "${BUILD_DIR}" -j"$(sysctl -n hw.ncpu)"

echo ""
find "${BUILD_DIR}" -name "_cpp*.so" -o -name "_cpp*.dylib" | head -5
