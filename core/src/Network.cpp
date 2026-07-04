#include "Network.hpp"

// =============================================================================
// Network Graph Compilation & Topological Pre-Allocation Pass
// =============================================================================
EngineStatus Network::compile(EngineContext& context, const std::vector<size_t>& input_shape) noexcept {
    if (m_layers.empty()) {
        context.log_error(StatusCode::Error_LayerInitializationFailed, 
            "Network compilation failed: Operational layer execution sequence is empty.");
        return EngineStatus{StatusCode::Error_LayerInitializationFailed, "No layers inside network graph."};
    }

    m_intermediary_buffers.clear();
    m_is_compiled = false;

    std::vector<size_t> current_shape = input_shape;
    size_t max_buffer_elements = 0;
    std::vector<size_t> largest_computed_shape;

    // Phase 1: Shape Trace and Trace Graph Dimensional Validity
    for (size_t i = 0; i < m_layers.size(); ++i) {
        std::vector<size_t> next_shape = m_layers[i]->compute_output_shape(current_shape);
        
        if (next_shape.empty()) {
            std::string err_msg = "Topological dimension breakdown at layer [" + 
                                  std::string(m_layers[i]->get_name()) + "] Index: " + std::to_string(i);
            context.log_error(StatusCode::Error_DimensionMismatch, err_msg);
            return EngineStatus{StatusCode::Error_DimensionMismatch, "Layer dimension mismatch during trace."};
        }

        // Evaluate overall aggregate elements to establish allocation scale boundaries
        size_t layer_element_count = 1;
        for (size_t dim : next_shape) {
            layer_element_count *= dim;
        }

        if (layer_element_count > max_buffer_elements) {
            max_buffer_elements = layer_element_count;
            largest_computed_shape = next_shape;
        }

        current_shape = next_shape; // Cascade shape down to subsequent layer node
    }

    // Phase 2: Ping-Pong Buffer Memory Allocation Phase
    // Allocate exactly two fixed-size tensors matching the peak required spatial footprint
    try {
        m_intermediary_buffers.reserve(2);
        m_intermediary_buffers.emplace_back(largest_computed_shape); // Buffer Track A
        m_intermediary_buffers.emplace_back(largest_computed_shape); // Buffer Track B
    } catch (...) {
        context.log_error(StatusCode::Error_MemoryAllocationFailed, 
            "Catastrophic failure initializing network workspace execution memory blocks.");
        return EngineStatus{StatusCode::Error_MemoryAllocationFailed, "Workspace pre-allocation failed."};
    }

    m_is_compiled = true;
    return EngineStatus{StatusCode::Success, "Network architecture compiled successfully with static allocations."};
}

// =============================================================================
// Low-Latency Forward Graph Execution Pipeline
// =============================================================================
EngineStatus Network::forward(EngineContext& context, const Tensor& input, Tensor& output) noexcept {
    if (!m_is_compiled) {
        context.log_error(StatusCode::Error_ExecutionFailure, 
            "Forward pass aborted: Engine graph has not been compiled or trace topology is corrupted.");
        return EngineStatus{StatusCode::Error_ExecutionFailure, "Network must be compiled prior to inference."};
    }

    const size_t total_layers = m_layers.size();

    // Edge-Case Handling: Optimization route for ultra-shallow single-layer topologies
    if (total_layers == 1) {
        return m_layers[0]->forward(context, input, output);
    }

    // Ping-Pong Pointer Tracking Logic Setup
    // track_in maps the target array to read from, track_out maps the target array to write into
    Tensor* track_in = &m_intermediary_buffers[0];
    Tensor* track_out = &m_intermediary_buffers[1];

    // Execute Initial Entry Layer using raw input source tensor
    EngineStatus status = m_layers[0]->forward(context, input, *track_in);
    if (!status.is_ok()) return status;

    // Core Intermediary Pipeline Execution Graph Loops
    for (size_t i = 1; i < total_layers - 1; ++i) {
        status = m_layers[i]->forward(context, *track_in, *track_out);
        if (!status.is_ok()) return status;

        // Perform Zero-Overhead Ping-Pong Pointer Swapping Pass
        std::swap(track_in, track_out);
    }

    // Execute Terminal Exit Layer routing evaluations straight to the target output destination buffer
    status = m_layers[total_layers - 1]->forward(context, *track_in, output);
    
    return status;
}