#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <new>
#include <cassert>

#ifndef ARENA_POOL_SIZE_BYTES
#define ARENA_POOL_SIZE_BYTES (2ULL * 1024 * 1024 * 1024)
#endif
#ifndef ARENA_CHUNK_SIZE
#define ARENA_CHUNK_SIZE (64 * 1024)
#endif
#ifndef ARENA_ALIGNMENT
#define ARENA_ALIGNMENT 64
#endif

#ifdef __APPLE__
#include <sys/mman.h>
#endif

namespace retrieval {

struct ArenaChunk {
    std::atomic<ArenaChunk*> next;
    std::atomic<std::size_t> offset;
    std::size_t capacity;
    alignas(ARENA_ALIGNMENT) char data[];
};

class ArenaPool {
public:
    static constexpr std::size_t POOL_SIZE = ARENA_POOL_SIZE_BYTES;
    static constexpr std::size_t CHUNK_SIZE = ARENA_CHUNK_SIZE;
    static constexpr std::size_t ALIGNMENT = ARENA_ALIGNMENT;

    ArenaPool() noexcept : head_(nullptr), total_allocated_(0) {
        head_ = allocate_chunk(CHUNK_SIZE);
        current_.store(head_, std::memory_order_relaxed);
    }
    ~ArenaPool() noexcept {
        ArenaChunk* chunk = head_;
        while (chunk) {
            ArenaChunk* next = chunk->next.load(std::memory_order_relaxed);
            deallocate_chunk(chunk);
            chunk = next;
        }
    }
    ArenaPool(const ArenaPool&) = delete;
    ArenaPool& operator=(const ArenaPool&) = delete;

    ArenaPool(ArenaPool&& other) noexcept
        : head_(other.head_), total_allocated_(other.total_allocated_.load(std::memory_order_relaxed)) {
        current_.store(other.current_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        other.head_ = nullptr;
        other.current_.store(nullptr, std::memory_order_relaxed);
    }
    ArenaPool& operator=(ArenaPool&& other) noexcept {
        if (this != &other) {
            this->~ArenaPool();
            head_ = other.head_;
            current_.store(other.current_.load(std::memory_order_relaxed), std::memory_order_relaxed);
            total_allocated_.store(other.total_allocated_.load(std::memory_order_relaxed), std::memory_order_relaxed);
            other.head_ = nullptr;
            other.current_.store(nullptr, std::memory_order_relaxed);
        }
        return *this;
    }

    [[nodiscard]] void* allocate(std::size_t size, std::size_t alignment = ALIGNMENT) noexcept {
        assert(alignment <= ALIGNMENT);
        assert(size > 0);
        const std::size_t aligned_size = align_up(size, alignment);
        ArenaChunk* chunk = current_.load(std::memory_order_acquire);
        while (chunk) {
            std::size_t offset = chunk->offset.load(std::memory_order_relaxed);
            std::size_t new_offset = align_up(offset, alignment) + aligned_size;
            if (new_offset <= chunk->capacity) {
                if (chunk->offset.compare_exchange_weak(offset, new_offset, std::memory_order_acq_rel)) {
                    total_allocated_.fetch_add(aligned_size, std::memory_order_relaxed);
                    return chunk->data + align_up(offset, alignment);
                }
            } else {
                ArenaChunk* next_chunk = chunk->next.load(std::memory_order_acquire);
                if (!next_chunk) {
                    std::size_t required_cap = (aligned_size > CHUNK_SIZE) ? aligned_size : CHUNK_SIZE;
                    next_chunk = allocate_chunk(required_cap);
                    ArenaChunk* expected = nullptr;
                    if (!chunk->next.compare_exchange_strong(expected, next_chunk, std::memory_order_acq_rel)) {
                        deallocate_chunk(next_chunk);
                        next_chunk = expected;
                    }
                }
                chunk = next_chunk;
                current_.store(chunk, std::memory_order_release);
            }
        }
        return nullptr;
    }

    template <typename T, typename... Args>
    [[nodiscard]] T* construct(Args&&... args) noexcept {
        void* ptr = allocate(sizeof(T), alignof(T));
        if (!ptr) return nullptr;
        return new (ptr) T(std::forward<Args>(args)...);
    }
    template <typename T>
    void destroy(T* ptr) noexcept { if (ptr) ptr->~T(); }

    void reset() noexcept {
        ArenaChunk* chunk = head_;
        while (chunk) {
            chunk->offset.store(0, std::memory_order_relaxed);
            chunk = chunk->next.load(std::memory_order_relaxed);
        }
        current_.store(head_, std::memory_order_release);
        total_allocated_.store(0, std::memory_order_relaxed);
    }

    [[nodiscard]] std::size_t total_allocated() const noexcept { return total_allocated_.load(std::memory_order_relaxed); }
    [[nodiscard]] std::size_t remaining() const noexcept {
        std::size_t used = total_allocated_.load(std::memory_order_relaxed);
        return (used < POOL_SIZE) ? (POOL_SIZE - used) : 0;
    }
    [[nodiscard]] bool exhausted() const noexcept { return total_allocated_.load(std::memory_order_relaxed) >= POOL_SIZE; }

private:
    static std::size_t align_up(std::size_t value, std::size_t alignment) noexcept {
        return (value + alignment - 1) & ~(alignment - 1);
    }
    static ArenaChunk* allocate_chunk(std::size_t capacity) noexcept {
        std::size_t total_size = sizeof(ArenaChunk) + capacity;
        void* mem = nullptr;
#ifdef __APPLE__
        mem = mmap(nullptr, total_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mem == MAP_FAILED) return nullptr;
#else
        mem = std::aligned_alloc(ARENA_ALIGNMENT, align_up(total_size, ARENA_ALIGNMENT));
        if (!mem) return nullptr;
#endif
        ArenaChunk* chunk = new (mem) ArenaChunk();
        chunk->next.store(nullptr, std::memory_order_relaxed);
        chunk->offset.store(0, std::memory_order_relaxed);
        chunk->capacity = capacity;
        return chunk;
    }
    static void deallocate_chunk(ArenaChunk* chunk) noexcept {
        if (!chunk) return;
        std::size_t total_size = sizeof(ArenaChunk) + chunk->capacity;
#ifdef __APPLE__
        munmap(chunk, align_up(total_size, 4096));
#else
        std::free(chunk);
#endif
    }
    ArenaChunk* head_;
    std::atomic<ArenaChunk*> current_;
    std::atomic<std::size_t> total_allocated_;
};

class ThreadLocalArena {
public:
    ThreadLocalArena() noexcept : pool_(nullptr) {}
    explicit ThreadLocalArena(ArenaPool* pool) noexcept : pool_(pool) {}
    ~ThreadLocalArena() = default;
    ThreadLocalArena(const ThreadLocalArena&) = delete;
    ThreadLocalArena& operator=(const ThreadLocalArena&) = delete;
    ThreadLocalArena(ThreadLocalArena&& other) noexcept : pool_(other.pool_) { other.pool_ = nullptr; }
    ThreadLocalArena& operator=(ThreadLocalArena&& other) noexcept {
        pool_ = other.pool_;
        other.pool_ = nullptr;
        return *this;
    }
    [[nodiscard]] void* allocate(std::size_t size, std::size_t alignment = ArenaPool::ALIGNMENT) noexcept {
        return pool_ ? pool_->allocate(size, alignment) : nullptr;
    }
    template <typename T, typename... Args>
    [[nodiscard]] T* construct(Args&&... args) noexcept {
        return pool_ ? pool_->construct<T>(std::forward<Args>(args)...) : nullptr;
    }
    template <typename T>
    void destroy(T* ptr) noexcept { if (pool_) pool_->destroy(ptr); }
    void reset() noexcept { if (pool_) pool_->reset(); }
    [[nodiscard]] bool exhausted() const noexcept { return pool_ && pool_->exhausted(); }
    [[nodiscard]] std::size_t remaining() const noexcept { return pool_ ? pool_->remaining() : 0; }
private:
    ArenaPool* pool_;
};

template <typename T, std::size_t N>
class FixedArray {
public:
    using value_type = T;
    using size_type = std::size_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;

    FixedArray() noexcept = default;
    ~FixedArray() noexcept = default;
    FixedArray(const FixedArray& other) noexcept { for (size_type i = 0; i < N; ++i) data_[i] = other.data_[i]; }
    FixedArray& operator=(const FixedArray& other) noexcept {
        for (size_type i = 0; i < N; ++i) data_[i] = other.data_[i];
        return *this;
    }
    FixedArray(FixedArray&& other) noexcept { for (size_type i = 0; i < N; ++i) data_[i] = std::move(other.data_[i]); }
    FixedArray& operator=(FixedArray&& other) noexcept {
        for (size_type i = 0; i < N; ++i) data_[i] = std::move(other.data_[i]);
        return *this;
    }
    [[nodiscard]] pointer data() noexcept { return data_; }
    [[nodiscard]] const_pointer data() const noexcept { return data_; }
    [[nodiscard]] reference operator[](size_type pos) noexcept { return data_[pos]; }
    [[nodiscard]] const_reference operator[](size_type pos) const noexcept { return data_[pos]; }
    [[nodiscard]] constexpr size_type size() const noexcept { return N; }
    [[nodiscard]] constexpr size_type max_size() const noexcept { return N; }
private:
    T data_[N];
};

namespace memory {
    using ArenaPool = retrieval::ArenaPool;
    using ThreadLocalArena = retrieval::ThreadLocalArena;
}
}  // namespace retrieval