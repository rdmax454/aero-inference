#pragma once

#include <vector>
#include <cstddef>
#include <memory>
#include <numeric>
#include <cassert>
#include "EngineContext.hpp"

/**
 * @brief High-Performance N-Dimensional Memory Matrix Abstraction Wrapper.
 * Manages cache-aligned flat allocation structures with zero heap modifications during inference.
 */
class Tensor {
private:
    std::vector<size_t> m_shape;
    std::vector<size_t> m_strides;
    std::unique_ptr<float[]> m_data{nullptr};
    size_t m_total_elements{0};
    size_t m_capacity{0}; // Actual allocated buffer size — never shrinks via reshape_view

    void compute_strides() noexcept {
        m_strides.resize(m_shape.size());
        if (m_shape.empty()) return;

        size_t current_stride = 1;
        for (int64_t i = static_cast<int64_t>(m_shape.size()) - 1; i >= 0; --i) {
            m_strides[i] = current_stride;
            current_stride *= m_shape[i];
        }
    }

public:
    Tensor() noexcept = default;
    ~Tensor() = default;

    explicit Tensor(const std::vector<size_t>& shape) : m_shape(shape) {
        if (m_shape.empty()) {
            return;
        }

        m_total_elements = std::accumulate(m_shape.begin(), m_shape.end(), size_t{1}, std::multiplies<size_t>());
        m_capacity = m_total_elements;

        m_data = std::make_unique<float[]>(m_total_elements);
        std::fill(m_data.get(), m_data.get() + m_total_elements, 0.0f);

        compute_strides();
    }

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&& other) noexcept
        : m_shape(std::move(other.m_shape)),
          m_strides(std::move(other.m_strides)),
          m_data(std::move(other.m_data)),
          m_total_elements(other.m_total_elements),
          m_capacity(other.m_capacity) {
        other.m_total_elements = 0;
        other.m_capacity = 0;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            m_shape = std::move(other.m_shape);
            m_strides = std::move(other.m_strides);
            m_data = std::move(other.m_data);
            m_total_elements = other.m_total_elements;
            m_capacity = other.m_capacity;
            other.m_total_elements = 0;
            other.m_capacity = 0;
        }
        return *this;
    }

    [[nodiscard]] inline float& operator[](size_t linear_index) noexcept {
#ifndef NDEBUG
        assert(linear_index < m_total_elements && "Out of bounds tensor linear index mutation requested.");
#endif
        return m_data[linear_index];
    }

    [[nodiscard]] inline const float& operator[](size_t linear_index) const noexcept {
#ifndef NDEBUG
        assert(linear_index < m_total_elements && "Out of bounds tensor linear index access requested.");
#endif
        return m_data[linear_index];
    }

    [[nodiscard]] inline size_t calculate_flat_offset(const std::vector<size_t>& coordinates) const noexcept {
#ifndef NDEBUG
        assert(coordinates.size() == m_shape.size() && "Coordinate dimension mismatch.");
#endif
        size_t flat_offset = 0;
        for (size_t i = 0; i < coordinates.size(); ++i) {
#ifndef NDEBUG
            assert(coordinates[i] < m_shape[i] && "Coordinate target overflows specific shape boundaries.");
#endif
            flat_offset += coordinates[i] * m_strides[i];
        }
        return flat_offset;
    }

    /**
     * @brief Re-labels this Tensor's logical shape WITHOUT reallocating memory.
     * Used by Network::forward() to correctly resize a reused ping-pong buffer
     * to match each layer's actual output shape, since buffers are pre-allocated
     * once at the size of the LARGEST layer output in the graph.
     * @param new_shape The real shape this buffer should represent right now.
     */
    void reshape_view(const std::vector<size_t>& new_shape) noexcept {
        size_t new_total = std::accumulate(new_shape.begin(), new_shape.end(), size_t{1}, std::multiplies<size_t>());
#ifndef NDEBUG
        assert(new_total <= m_capacity && "reshape_view: new shape exceeds allocated buffer capacity.");
#endif
        m_shape = new_shape;
        m_total_elements = new_total;
        compute_strides();
    }

    [[nodiscard]] size_t size() const noexcept { return m_total_elements; }
    [[nodiscard]] const std::vector<size_t>& shape() const noexcept { return m_shape; }
    [[nodiscard]] float* data() noexcept { return m_data.get(); }
    [[nodiscard]] const float* data() const noexcept { return m_data.get(); }
};