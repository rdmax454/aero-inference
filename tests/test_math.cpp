#include "MathBackend.hpp"
#include <iostream>
#include <cmath>
#include <cstddef>

bool nearly_equal(float a, float b, float epsilon = 0.0001f) {
    return std::fabs(a - b) < epsilon;
}

// =============================================================================
// Test 1: ReLU Activation
// =============================================================================
bool test_relu_activation() {
    float data[] = {-5.0f, 2.0f, 0.0f};
    float expected[] = {0.0f, 2.0f, 0.0f};
    size_t size = 3;

    MathBackend::relu_activation(data, size);

    for (size_t i = 0; i < size; ++i) {
        if (!nearly_equal(data[i], expected[i])) return false;
    }
    return true;
}

// =============================================================================
// Test 2: Fully Connected Forward
// Hand-calculated: output[o] = bias[o] + sum(input[i] * weights[i*out_dim+o])
// =============================================================================
bool test_fully_connected_forward() {
    float input[] = {1.0f, 2.0f};
    float weights[] = {1.0f, 0.0f,
                        0.0f, 1.0f};
    float bias[] = {0.5f, 0.5f};
    float output[2] = {0.0f, 0.0f};
    float expected[] = {1.5f, 2.5f};

    size_t input_dim = 2, output_dim = 2;

    EngineStatus status = MathBackend::fully_connected_forward(
        input, weights, bias, output, input_dim, output_dim
    );

    if (!status.is_ok()) return false;

    for (size_t i = 0; i < output_dim; ++i) {
        if (!nearly_equal(output[i], expected[i])) return false;
    }
    return true;
}

// =============================================================================
// Test 3: Conv2D Forward
// Input 3x3 (single channel):        Kernel 2x2:
//   1 2 3                              1 0
//   4 5 6                              0 1
//   7 8 9
// No padding, stride 1 -> output is 2x2.
// Hand-calculated (cross-correlation, top-left aligned):
//   out[0][0] = 1*1 + 2*0 + 4*0 + 5*1 = 6
//   out[0][1] = 2*1 + 3*0 + 5*0 + 6*1 = 8
//   out[1][0] = 4*1 + 5*0 + 7*0 + 8*1 = 12
//   out[1][1] = 5*1 + 6*0 + 8*0 + 9*1 = 14
// =============================================================================
bool test_conv2d_forward() {
    float input[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    };
    float kernel[] = {
        1.0f, 0.0f,
        0.0f, 1.0f
    };
    float output[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float expected[] = {6.0f, 8.0f, 12.0f, 14.0f};

    EngineStatus status = MathBackend::conv2d_forward(
        input, kernel, output,
        /*in_channels=*/1, /*out_channels=*/1,
        /*in_height=*/3, /*in_width=*/3,
        /*kernel_size=*/2, /*stride=*/1, /*padding=*/0
    );

    if (!status.is_ok()) return false;

    for (size_t i = 0; i < 4; ++i) {
        if (!nearly_equal(output[i], expected[i])) return false;
    }
    return true;
}

// =============================================================================
// Test 4: Edge Case — Null Pointer Handling
// A well-behaved engine must return an error status, not crash, on bad input.
// =============================================================================
bool test_fully_connected_null_pointer() {
    float output[2] = {0.0f, 0.0f};

    EngineStatus status = MathBackend::fully_connected_forward(
        nullptr, nullptr, nullptr, output, 2, 2
    );

    // We EXPECT this to fail gracefully, not crash.
    return !status.is_ok();
}

bool test_conv2d_null_pointer() {
    float output[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    EngineStatus status = MathBackend::conv2d_forward(
        nullptr, nullptr, output, 1, 1, 3, 3, 2, 1, 0
    );

    return !status.is_ok();
}

// =============================================================================
// Test 5: Edge Case — Zero-Size Input
// =============================================================================
bool test_relu_zero_size() {
    float data[] = {1.0f}; // array must exist, but we tell the function size 0
    MathBackend::relu_activation(data, 0);
    // If this line is reached without crashing, the test passes.
    return true;
}

int main() {
    int passed = 0;
    int total = 0;

    auto run = [&](const char* name, bool (*test_fn)()) {
        total++;
        if (test_fn()) {
            std::cout << "[PASS] " << name << "\n";
            passed++;
        } else {
            std::cout << "[FAIL] " << name << "\n";
        }
    };

    run("test_relu_activation", test_relu_activation);
    run("test_fully_connected_forward", test_fully_connected_forward);
    run("test_conv2d_forward", test_conv2d_forward);
    run("test_fully_connected_null_pointer", test_fully_connected_null_pointer);
    run("test_conv2d_null_pointer", test_conv2d_null_pointer);
    run("test_relu_zero_size", test_relu_zero_size);

    std::cout << "\n" << passed << "/" << total << " tests passed.\n";
    return (passed == total) ? 0 : 1;
}