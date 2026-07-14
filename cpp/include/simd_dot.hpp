#pragma once

#include <cstdint>
#include <cstddef>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define HAS_NEON 1
#else
#define HAS_NEON 0
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

#if defined(__AVX2__)
#define HAS_AVX2 1
#else
#define HAS_AVX2 0
#endif

namespace retrieval {
namespace simd {

static constexpr std::size_t EMBEDDING_DIM = 128;

[[gnu::always_inline]]
inline float dot_scalar(const float* __restrict a, const float* __restrict b, std::size_t n) noexcept {
    float result = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

#if HAS_NEON
[[gnu::always_inline]]
inline float dot_neon(const float* __restrict a, const float* __restrict b, std::size_t n) noexcept {
    float32x4_t sum = vdupq_n_f32(0.0f);
    std::size_t i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        sum = vfmaq_f32(sum, va, vb);
    }
    float result = vaddvq_f32(sum);
    for (; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
}
#endif

#if HAS_AVX2
[[gnu::always_inline]]
inline float dot_avx2(const float* __restrict a, const float* __restrict b, std::size_t n) noexcept {
    __m256 sum = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    float result_arr[8];
    _mm256_storeu_ps(result_arr, sum);
    float total = result_arr[0] + result_arr[1] + result_arr[2] + result_arr[3] + 
                  result_arr[4] + result_arr[5] + result_arr[6] + result_arr[7];
    for (; i < n; ++i) {
        total += a[i] * b[i];
    }
    return total;
}
#endif

[[gnu::always_inline]]
inline float dot_product(const float* __restrict a, const float* __restrict b, std::size_t n) noexcept {
#if HAS_NEON
    return dot_neon(a, b, n);
#elif HAS_AVX2
    return dot_avx2(a, b, n);
#else
    return dot_scalar(a, b, n);
#endif
}

}  // namespace simd
}  // namespace retrieval
