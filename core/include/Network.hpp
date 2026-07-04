#pragma once

#include <vector>
#include <memory>
#include <string>
#include "layers/ILayer.hpp"
#include "Tensor.hpp"
#include "EngineContext.hpp"

/**
 * @brief The Graph Execution Orchestrator managing structural network layers.
 * Performs topological shape validation passes and handles zero-allocation inference graphs.
 */
class Network {
private:
    std::vector<std::unique_ptr<ILayer>> m_layers;
    std::vector<Tensor> m_intermediary_buffers; // Reusable internal memory allocations
    bool m_is_compiled{false};

public:
    Network() noexcept = default;
    ~Network() = default;

    // Prohibit network duplicating to defend contiguous memory footprints
    Network(const Network&) = delete;
    Network& operator=(const Network&) = delete;

    // Support clean zero-copy graph transitions
    Network(Network&&) noexcept = default;
    Network& operator=(Network&&) noexcept = default;

    /**
     * @brief Appends a structural computational layer to the model pipeline tail.
     * @param layer A unique_ptr owning the specific target layer interface.
     */
    void add_layer(std::unique_ptr<ILayer> layer) noexcept {
        if (layer) {
            m_layers.push_back(std::move(layer));
            m_is_compiled = false; // Model altered, must re-verify boundaries
        }
    }

    /**
     * @brief Traverses the execution pipeline to check shape boundaries and allocate intermediate buffers.
     * @param context Global runtime engine safety manager.
     * @param input_shape The precise geometric shape dimensions expected at input entry.
     * @return EngineStatus Tracking structure proving network architectural soundness.
     */
    EngineStatus compile(EngineContext& context, const std::vector<size_t>& input_shape) noexcept;

    /**
     * @brief Executes high-speed forward inference across the calculated network graph.
     * @param context Global runtime engine safety manager.
     * @param input Direct incoming raw feed data wrapper.
     * @param output Destination tensor tracking computation outcomes.
     * @return EngineStatus Evaluation results marking forward execution accuracy.
     */
    EngineStatus forward(EngineContext& context, const Tensor& input, Tensor& output) noexcept;

    /**
     * @brief Evaluates whether the compilation structure is currently locked down and executable.
     */
    [[nodiscard]] bool is_compiled() const noexcept { return m_is_compiled; }

    /**
     * @brief Clears all internal graph architecture layouts and flushes layer sequences.
     */
    void clear() noexcept {
        m_layers.clear();
        m_intermediary_buffers.clear();
        m_is_compiled = false;
    }

    /**
     * @brief Utility returning total active structural nodes mapped inside the engine.
     */
    [[nodiscard]] size_t layer_count() const noexcept { return m_layers.size(); }
};