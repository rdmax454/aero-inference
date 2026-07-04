#include "MathBackend.hpp"
#include <algorithm>
#include <cmath>

// =============================================================================
// Fully Connected (Dense) Layer Kernel
// =============================================================================
EngineStatus MathBackend::fully_connected_forward(
    const float* __restrict input,
    const float* __restrict weights,
    const float* __restrict bias,
    float* __restrict output,
    size_t input_dim,
    size_t output_dim
) noexcept {
    if (!input || !weights || !bias || !output) {
        return EngineStatus{StatusCode::Error_NullPointer, "Null pointer passed to Fully Connected execution kernel."};
    }

    // Initialize output with bias values to avoid a separate broadcast pass
    for (size_t o = 0; o < output_dim; ++o) {
        output[o] = bias[o];
    }

    // Optimized Core Matrix Multiplication Loop (Input-Major Cache Locality Pattern)
    // By keeping the input loop outermost, we stream sequentially through the weight matrix
    for (size_t i = 0; i < input_dim; ++i) {
        const float input_val = input[i];
        const size_t weight_row_offset = i * output_dim;

        for (size_t o = 0; o < output_dim; ++o) {
            output[o] += input_val * weights[weight_row_offset + o];
        }
    }

    return EngineStatus{StatusCode::Success, "Fully connected forward pass completed."};
}

// =============================================================================
// ReLU Activation Kernel
// =============================================================================
void MathBackend::relu_activation(float* data, size_t size) noexcept {
    if (!data) return;

    // Loop optimized for automatic compiler SIMD vectorization (AVX2/NEON)
    for (size_t i = 0; i < size; ++i) {
        data[i] = std::max(0.0f, data[i]);
    }
}

// =============================================================================
// 2D Convolution / Cross-Correlation Kernel
// =============================================================================
EngineStatus MathBackend::conv2d_forward(
    const float* __restrict input,
    const float* __restrict kernel,
    float* __restrict output,
    size_t in_channels,
    size_t out_channels,
    size_t in_height,
    size_t in_width,
    size_t kernel_size,
    size_t stride,
    size_t padding
) noexcept {
    if (!input || !kernel || !output) {
        return EngineStatus{StatusCode::Error_NullPointer, "Null pointer passed to Conv2D execution kernel."};
    }

    // Compute explicit output spatial dimensions
    const size_t out_height = ((in_height + 2 * padding - kernel_size) / stride) + 1;
    const size_t out_width  = ((in_width + 2 * padding - kernel_size) / stride) + 1;

    // Zero out output buffer entirely before execution accumulations
    const size_t total_output_elements = out_channels * out_height * out_width;
    std::fill(output, output + total_output_elements, 0.0f);

    // Deep Industrial-Grade 6D Loop Pipeline
    for (size_t oc = 0; oc < out_channels; ++oc) {
        const size_t out_channel_offset = oc * out_height * out_width;
        const size_t kernel_channel_offset = oc * in_channels * kernel_size * kernel_size;

        for (size_t oh = 0; oh < out_height; ++oh) {
            // Map output coordinate space back to padded input coordinate space
            const int64_t in_y_origin = static_cast<int64_t>(oh * stride) - static_cast<int64_t>(padding);

            for (size_t ow = 0; ow < out_width; ++ow) {
                const int64_t in_x_origin = static_cast<int64_t>(ow * stride) - static_cast<int64_t>(padding);
                
                float accumulator = 0.0f;

                for (size_t ic = 0; ic < in_channels; ++ic) {
                    const size_t input_channel_offset = ic * in_height * in_width;
                    const size_t kernel_filter_offset = kernel_channel_offset + (ic * kernel_size * kernel_size);

                    for (size_t ky = 0; ky < kernel_size; ++ky) {
                        const int64_t current_y = in_y_origin + static_cast<int64_t>(ky);
                        
                        // Bound guard: Skip calculation loop if inside the top/bottom padded virtual zeros zone
                        if (current_y < 0 || current_y >= static_cast<int64_t>(in_height)) continue;

                        const size_t input_row_offset = input_channel_offset + (static_cast<size_t>(current_y) * in_width);
                        const size_t kernel_row_offset = kernel_filter_offset + (ky * kernel_size);

                        for (size_t kx = 0; kx < kernel_size; ++kx) {
                            const int64_t current_x = in_x_origin + static_cast<int64_t>(kx);

                            // Bound guard: Skip calculation loop if inside the left/right padded virtual zeros zone
                            if (current_x < 0 || current_x >= static_cast<int64_t>(in_width)) continue;

                            const float pixels = input[input_row_offset + static_cast<size_t>(current_x)];
                            const float weight = kernel[kernel_row_offset + kx];

                            accumulator += pixels * weight;
                        }
                    }
                }
                
                // Write output back to memory flat spatial coordinates
                output[out_channel_offset + (oh * out_width) + ow] = accumulator;
            }
        }
    }

    return EngineStatus{StatusCode::Success, "Conv2D forward pass completed successfully."};
}