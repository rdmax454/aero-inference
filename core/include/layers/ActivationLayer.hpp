#pragma once

#include "ILayer.hpp"
#include "../MathBackend.hpp"
#include <string_view>
#include <utility>

/**
 * @brief Standalone Activation Computational Node Layer.
 * Executes element-wise nonlinear activation passes (such as ReLU) across entire tensors.
 */
class ActivationLayer final : public ILayer {
private:
    std::string m_name;
    LayerType m_type; // Allows configuration of different activation types if expanded later

public:
    /**
     * @brief Constructor initializing the activation layer node.
     * @param name Unique tracking identifier for logging or GUI tracking dashboards.
     */
    explicit ActivationLayer(std::string name) noexcept
        : m_name(std::move(name)), m_type(LayerType::Activation_ReLU) {}

    ~ActivationLayer() override = default;

    // Enforce strict asset ownership: ban duplication paths
    ActivationLayer(const ActivationLayer&) = delete;
    ActivationLayer& operator=(const ActivationLayer&) = delete;
    ActivationLayer(ActivationLayer&&) noexcept = default;
    ActivationLayer& operator=(ActivationLayer&&) noexcept = default;

    /**
     * @brief Executes high-speed, in-place mathematical activation using optimized math backends.
     */
    EngineStatus forward(EngineContext& context, const Tensor& input, Tensor& output) noexcept override {
        (void)context; // Context unreferenced here since in-place vector math can't trigger shape breaks

        // Professional optimization: If input and output are different buffers, copy first.
        // If they are part of our network's ping-pong pool, write straight to output.
        if (input.data() != output.data()) {
            std::copy(input.data(), input.data() + input.size(), output.data());
        }

        // Direct hardware acceleration execution step via flat sequential pointers
        MathBackend::relu_activation(output.data(), output.size());

        return EngineStatus{StatusCode::Success, "ReLU activation pass completed successfully."};
    }

    /**
     * @brief Activation functions are purely element-wise operations. 
     * The output shape exactly matches the input shape with zero spatial changes.
     */
    [[nodiscard]] std::vector<size_t> compute_output_shape(const std::vector<size_t>& input_shape) const noexcept override {
        return input_shape; // Complete 1:1 structural passthrough
    }

    // System Utilities and Reflection Interfaces
    [[nodiscard]] LayerType get_type() const noexcept override { 
        return m_type; 
    }
    
    [[nodiscard]] std::string_view get_name() const noexcept override { 
        return m_name; 
    }
};