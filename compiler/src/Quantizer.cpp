#include "Quantizer.hpp"
#include <cmath>
#include <algorithm>

// =============================================================================
// Symmetric Max-Absolute Calibration Engine Pass
// =============================================================================
EngineStatus Quantizer::calibrate_symmetric(
    EngineContext& context,
    const Tensor& reference_tensor
) noexcept {
    const size_t element_count = reference_tensor.size();
    if (element_count == 0) {
        context.log_error(StatusCode::Error_DimensionMismatch, 
            "Quantization calibration aborted: Source reference data tensor is empty.");
        return EngineStatus{StatusCode::Error_DimensionMismatch, "Empty tensor passed for calibration."};
    }

    const float* raw_data = reference_tensor.data();
    float max_absolute_value = 0.0f;

    // Scan the entire tensor grid to isolate the peak dynamic magnitude
    for (size_t i = 0; i < element_count; ++i) {
        const float absolute_val = std::abs(raw_data[i]);
        if (absolute_val > max_absolute_value) {
            max_absolute_value = absolute_val;
        }
    }

    // Edge Case Protection: Handle dead or uniformly zero-initialized parameter tensors safely
    if (max_absolute_value < 1e-7f) {
        m_scale = 1.0f;
        m_zero_point = 0;
        m_is_calibrated = true;
        return EngineStatus{StatusCode::Success, "Symmetric calibration complete (Uniform dead zero matrix identified)."};
    }

    // Compute the exact linear mapping ratio down to the INT8 maximum limit boundary (127)
    // Math Formula: Scale = Max(Absolute(Value)) / 127.0
    m_scale = max_absolute_value / 127.0f;
    m_zero_point = 0; // Symmetric mapping locks the center zero-point offset precisely to 0
    m_is_calibrated = true;

    return EngineStatus{StatusCode::Success, "Symmetric dynamic quantization calibration calculated successfully."};
}

// =============================================================================
// High-Performance Quantization & Simulated Compression Pass
// =============================================================================
EngineStatus Quantizer::quantize_tensor(
    EngineContext& context,
    const Tensor& input,
    Tensor& output
) const noexcept {
    if (!m_is_calibrated) {
        context.log_error(StatusCode::Error_ExecutionFailure, 
            "Quantization execution halted: Calibration pass must occur prior to compressing parameters.");
        return EngineStatus{StatusCode::Error_ExecutionFailure, "Quantizer utility is uncalibrated."};
    }

    const size_t total_elements = input.size();
    if (total_elements != output.size()) {
        context.log_error(StatusCode::Error_DimensionMismatch, 
            "Quantization transformation aborted: Target output buffer size does not match source input tensor.");
        return EngineStatus{StatusCode::Error_DimensionMismatch, "Tensor allocation bounds mismatch."};
    }

    const float* __restrict src = input.data();
    float* __restrict dest = output.data();
    const float inverse_scale = 1.0f / m_scale;

    // Loop optimized for seamless CPU register pipelining and auto-vectorization
    for (size_t i = 0; i < total_elements; ++i) {
        // Step 1: Scale raw FP32 data into int8 space using our pre-calculated ratio
        float scaled_value = src[i] * inverse_scale;

        // Step 2: Use round-to-nearest mathematical conventions
        float rounded_value = std::round(scaled_value);

        // Step 3: Hard-clip values to defend against 8-bit signed integer register overflow
        float clipped_value = std::max(-128.0f, std::min(127.0f, rounded_value));

        // Step 4: De-quantize immediately to execute an industrial Fake-Quantization simulation
        dest[i] = clipped_value * m_scale;
    }

    return EngineStatus{StatusCode::Success, "Tensor quantization transformation pipeline executed successfully."};
}