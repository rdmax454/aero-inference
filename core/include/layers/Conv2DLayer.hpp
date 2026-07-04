#pragma once

#include "ILayer.hpp"
#include "../MathBackend.hpp"
#include <string_view>
#include <utility>

/**
 * @brief Industrial-Grade 2D Spatial Convolutional Computational Layer.
 * Manages spatial feature maps, stride factors, padding boundaries, and weight kernels.
 */
class Conv2DLayer final : public ILayer {
private:
    size_t m_in_channels{0};
    size_t m_out_channels{0};
    size_t m_kernel_size{0};
    size_t m_stride{1};
    size_t m_padding{0};

    // Weight parameters: Packed flat structure expected shape: {out_channels, in_channels, kernel_size, kernel_size}
    Tensor m_kernel_weights;
    std::string m_name;

public:
    /**
     * @brief Constructor initializing convolutional parameters and moving ownership of kernel weights.
     * @param name Unique tracking nomenclature identifier for logging or GUI tracking.
     * @param in_channels The feature depth count of incoming data blocks.
     * @param out_channels The total volume of feature filters to apply.
     * @param kernel_size Structural width/height boundary of the spatial filter (assumed square).
     * @param stride Stepping translation offset factor across input grids.
     * @param padding Virtual edge pixel expansion size to append around the boundaries.
     * @param weights Parameter tensor containing pre-trained convolution kernel weights.
     */
    Conv2DLayer(std::string name, size_t in_channels, size_t out_channels, size_t kernel_size, 
                size_t stride, size_t padding, Tensor&& weights) noexcept
        : m_in_channels(in_channels),
          m_out_channels(out_channels),
          m_kernel_size(kernel_size),
          m_stride(stride),
          m_padding(padding),
          m_kernel_weights(std::move(weights)),
          m_name(std::move(name)) {}

    ~Conv2DLayer() override = default;

    // Enforce strict asset ownership: ban duplication paths
    Conv2DLayer(const Conv2DLayer&) = delete;
    Conv2DLayer& operator=(const Conv2DLayer&) = delete;
    Conv2DLayer(Conv2DLayer&&) noexcept = default;
    Conv2DLayer& operator=(Conv2DLayer&&) noexcept = default;

    /**
     * @brief Triggers the hardware-optimized Conv2D spatial execution pass.
     */
    EngineStatus forward(EngineContext& context, const Tensor& input, Tensor& output) noexcept override {
        const std::vector<size_t>& in_shape = input.shape();
        
        // Runtime Verification: Enforced during internal debug profiling passes
#ifndef NDEBUG
        if (in_shape.size() != 3) {
            context.log_error(StatusCode::Error_DimensionMismatch, 
                "Conv2DLayer [" + m_name + "] Execution error: Input tensor must exhibit a 3D structural layout {C, H, W}.");
            return EngineStatus{StatusCode::Error_DimensionMismatch, "Invalid tensor dimension topology."};
        }
        if (in_shape[0] != m_in_channels) {
            context.log_error(StatusCode::Error_DimensionMismatch, 
                "Conv2DLayer [" + m_name + "] Execution error: Input channel depth does not match structural constraints.");
            return EngineStatus{StatusCode::Error_DimensionMismatch, "Channel dimensionality mismatch."};
        }
#endif

        const size_t in_height = in_shape[1];
        const size_t in_width  = in_shape[2];

        // Route directly to our highly localized spatial cross-correlation math module
        return MathBackend::conv2d_forward(
            input.data(),
            m_kernel_weights.data(),
            output.data(),
            m_in_channels,
            m_out_channels,
            in_height,
            in_width,
            m_kernel_size,
            m_stride,
            m_padding
        );
    }

    /**
     * @brief Analyzes input image maps to evaluate structural boundaries and calculate the output tensor geometry.
     * Mathematical Formula: Out = ((In + 2*Pad - Kernel) / Stride) + 1
     */
    [[nodiscard]] std::vector<size_t> compute_output_shape(const std::vector<size_t>& input_shape) const noexcept override {
        if (input_shape.size() != 3 || input_shape[0] != m_in_channels) {
            return {}; // Returning empty signals an initialization validation breakdown
        }

        const size_t in_height = input_shape[1];
        const size_t in_width  = input_shape[2];

        // Ensure calculations do not result underflow constraints
        if ((in_height + 2 * m_padding) < m_kernel_size || (in_width + 2 * m_padding) < m_kernel_size) {
            return {};
        }

        const size_t out_height = ((in_height + 2 * m_padding - m_kernel_size) / m_stride) + 1;
        const size_t out_width  = ((in_width + 2 * m_padding - m_kernel_size) / m_stride) + 1;

        // Returns explicit multi-dimensional shape mapping {Out_Channels, Out_Height, Out_Width}
        return { m_out_channels, out_height, out_width };
    }

    // System Utilities and Reflection Interfaces
    [[nodiscard]] LayerType get_type() const noexcept override { 
        return LayerType::Conv2D; 
    }
    
    [[nodiscard]] std::string_view get_name() const noexcept override { 
        return m_name; 
    }

    [[nodiscard]] size_t in_channels() const noexcept { return m_in_channels; }
    [[nodiscard]] size_t out_channels() const noexcept { return m_out_channels; }
    [[nodiscard]] size_t kernel_size() const noexcept { return m_kernel_size; }
    [[nodiscard]] size_t stride() const noexcept { return m_stride; }
    [[nodiscard]] size_t padding() const noexcept { return m_padding; }
    [[nodiscard]] const Tensor& kernel_weights() const noexcept { return m_kernel_weights; }
};