#include "Quantizer.hpp"
#include "Tensor.hpp"
#include "EngineContext.hpp"
#include <cmath>
#include <iostream>

bool nearly_equal(float a, float b, float epsilon = 0.0001f) {
    return std::fabs(a - b) < epsilon;
}

// Calibration on {-2, 0, 1, 2}: max |x| = 2, so scale = 2 / 127.
bool test_calibrate_symmetric_scale() {
    EngineContext context;
    context.set_debug_mode(false);
    Quantizer q;

    Tensor reference({4});
    reference[0] = -2.0f; reference[1] = 0.0f; reference[2] = 1.0f; reference[3] = 2.0f;

    EngineStatus status = q.calibrate_symmetric(context, reference);
    if (!status.is_ok()) return false;
    if (!q.is_calibrated()) return false;
    if (q.zero_point() != 0) return false;
    return nearly_equal(q.scale(), 2.0f / 127.0f, 1e-6f);
}

// With scale = 2/127: -2 -> -127 -> -2.0, 0 -> 0, 0.5 -> 31.75 -> 32 -> 32*2/127, 2 -> 127 -> 2.0.
bool test_quantize_known_values() {
    EngineContext context;
    context.set_debug_mode(false);
    Quantizer q;

    Tensor input({4});
    input[0] = -2.0f; input[1] = 0.0f; input[2] = 0.5f; input[3] = 2.0f;
    if (!q.calibrate_symmetric(context, input).is_ok()) return false;

    Tensor output({4});
    if (!q.quantize_tensor(context, input, output).is_ok()) return false;

    const float expected[] = {-2.0f, 0.0f, 32.0f * 2.0f / 127.0f, 2.0f};
    for (size_t i = 0; i < 4; ++i) {
        if (!nearly_equal(output[i], expected[i])) return false;
    }
    return true;
}

// For values inside the calibrated range, rounding error can never exceed half a quantization step.
bool test_quantize_error_bound() {
    EngineContext context;
    context.set_debug_mode(false);
    Quantizer q;

    const size_t n = 150;
    Tensor input({n});
    for (size_t i = 0; i < n; ++i) input[i] = -1.0f + 0.0133f * static_cast<float>(i);
    if (!q.calibrate_symmetric(context, input).is_ok()) return false;

    Tensor output({n});
    if (!q.quantize_tensor(context, input, output).is_ok()) return false;

    const float bound = q.scale() / 2.0f + 1e-5f;
    for (size_t i = 0; i < n; ++i) {
        if (std::fabs(output[i] - input[i]) > bound) return false;
    }
    return true;
}

// An all-zero tensor must not divide by zero: scale falls back to 1 and values stay 0.
bool test_all_zero_tensor() {
    EngineContext context;
    context.set_debug_mode(false);
    Quantizer q;

    Tensor input({3});
    if (!q.calibrate_symmetric(context, input).is_ok()) return false;
    if (!nearly_equal(q.scale(), 1.0f)) return false;

    Tensor output({3});
    if (!q.quantize_tensor(context, input, output).is_ok()) return false;
    for (size_t i = 0; i < 3; ++i) {
        if (!nearly_equal(output[i], 0.0f)) return false;
    }
    return true;
}

// Values far above the calibrated range must clamp at +127 steps, not overflow.
bool test_upper_clamp() {
    EngineContext context;
    context.set_debug_mode(false);
    Quantizer q;
    q.force_parameters(0.1f, 0);

    Tensor input({1});
    input[0] = 100.0f;
    Tensor output({1});
    if (!q.quantize_tensor(context, input, output).is_ok()) return false;
    return nearly_equal(output[0], 12.7f);
}

// Expected failures: empty tensor, uncalibrated use, and size mismatch must return errors, not crash.
bool test_error_paths() {
    EngineContext context;
    context.set_debug_mode(false);

    Quantizer q;
    Tensor empty;
    if (q.calibrate_symmetric(context, empty).is_ok()) return false;

    Tensor input({2});
    Tensor output({2});
    if (q.quantize_tensor(context, input, output).is_ok()) return false;

    q.force_parameters(1.0f, 0);
    Tensor wrong_size({3});
    if (q.quantize_tensor(context, input, wrong_size).is_ok()) return false;

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

    run("test_calibrate_symmetric_scale", test_calibrate_symmetric_scale);
    run("test_quantize_known_values", test_quantize_known_values);
    run("test_quantize_error_bound", test_quantize_error_bound);
    run("test_all_zero_tensor", test_all_zero_tensor);
    run("test_upper_clamp", test_upper_clamp);
    run("test_error_paths", test_error_paths);

    std::cout << "\n" << passed << "/" << total << " tests passed.\n";
    return (passed == total) ? 0 : 1;
}