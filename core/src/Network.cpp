#include "Network.hpp"

EngineStatus Network::compile(EngineContext& context, const std::vector<size_t>& input_shape) noexcept {
    if (m_layers.empty()) {
        context.log_error(StatusCode::Error_LayerInitializationFailed,
            "Network compilation failed: Operational layer execution sequence is empty.");
        return EngineStatus{StatusCode::Error_LayerInitializationFailed, "No layers inside network graph."};
    }

    m_intermediary_buffers.clear();
    m_layer_output_shapes.clear();
    m_is_compiled = false;

    std::vector<size_t> current_shape = input_shape;
    size_t max_buffer_elements = 0;
    std::vector<size_t> largest_computed_shape;

    for (size_t i = 0; i < m_layers.size(); ++i) {
        std::vector<size_t> next_shape = m_layers[i]->compute_output_shape(current_shape);

        if (next_shape.empty()) {
            std::string err_msg = "Topological dimension breakdown at layer [" +
                                  std::string(m_layers[i]->get_name()) + "] Index: " + std::to_string(i);
            context.log_error(StatusCode::Error_DimensionMismatch, err_msg);
            return EngineStatus{StatusCode::Error_DimensionMismatch, "Layer dimension mismatch during trace."};
        }

        m_layer_output_shapes.push_back(next_shape);

        size_t layer_element_count = 1;
        for (size_t dim : next_shape) {
            layer_element_count *= dim;
        }

        if (layer_element_count > max_buffer_elements) {
            max_buffer_elements = layer_element_count;
            largest_computed_shape = next_shape;
        }

        current_shape = next_shape;
    }

    try {
        m_intermediary_buffers.reserve(2);
        m_intermediary_buffers.emplace_back(largest_computed_shape);
        m_intermediary_buffers.emplace_back(largest_computed_shape);
    } catch (...) {
        context.log_error(StatusCode::Error_MemoryAllocationFailed,
            "Catastrophic failure initializing network workspace execution memory blocks.");
        return EngineStatus{StatusCode::Error_MemoryAllocationFailed, "Workspace pre-allocation failed."};
    }

    m_is_compiled = true;
    return EngineStatus{StatusCode::Success, "Network architecture compiled successfully with static allocations."};
}

EngineStatus Network::forward(EngineContext& context, const Tensor& input, Tensor& output) noexcept {
    if (!m_is_compiled) {
        context.log_error(StatusCode::Error_ExecutionFailure,
            "Forward pass aborted: Engine graph has not been compiled or trace topology is corrupted.");
        return EngineStatus{StatusCode::Error_ExecutionFailure, "Network must be compiled prior to inference."};
    }

    const size_t total_layers = m_layers.size();

    if (total_layers == 1) {
        return m_layers[0]->forward(context, input, output);
    }

    Tensor* track_in = &m_intermediary_buffers[0];
    Tensor* track_out = &m_intermediary_buffers[1];

    // Relabel track_in to layer 0's REAL output shape before it receives data.
    track_in->reshape_view(m_layer_output_shapes[0]);
    EngineStatus status = m_layers[0]->forward(context, input, *track_in);
    if (!status.is_ok()) return status;

    for (size_t i = 1; i < total_layers - 1; ++i) {
        // Relabel track_out to THIS layer's real output shape before writing into it.
        track_out->reshape_view(m_layer_output_shapes[i]);
        status = m_layers[i]->forward(context, *track_in, *track_out);
        if (!status.is_ok()) return status;

        std::swap(track_in, track_out);
    }

    status = m_layers[total_layers - 1]->forward(context, *track_in, output);

    return status;
}