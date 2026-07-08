#include "Network.hpp"
#include "layers/DenseLayer.hpp"
#include "EngineContext.hpp"
#include <iostream>
#include <cmath>
#include <memory>

bool nearly_equal(float a, float b, float epsilon = 0.0001f) {
    return std::fabs(a - b) < epsilon;
}

// =============================================================================
// Test 1: End-to-end compile + forward through a single DenseLayer
// Reuses the exact same known-correct math as test_fully_connected_forward,
// but now proves the FULL pipeline (Network -> Layer -> MathBackend) agrees.
// =============================================================================
bool test_network_single_dense_layer() {
    EngineContext context;
    Network network;

    Tensor weights({2, 2});
    weights[0] = 1.0f; weights[1] = 0.0f;
    weights[2] = 0.0f; weights[3] = 1.0f;

    Tensor bias({2});
    bias[0] = 0.5f; bias[1] = 0.5f;

    network.add_layer(std::make_unique<DenseLayer>(
        "test_dense", 2, 2, std::move(weights), std::move(bias)
    ));

    EngineStatus compile_status = network.compile(context, {2});
    if (!compile_status.is_ok()) return false;
    if (!network.is_compiled()) return false;

    Tensor input({2});
    input[0] = 1.0f;
    input[1] = 2.0f;

    Tensor output({2});
    EngineStatus forward_status = network.forward(context, input, output);
    if (!forward_status.is_ok()) return false;

    float expected[] = {1.5f, 2.5f};
    if (!nearly_equal(output[0], expected[0])) return false;
    if (!nearly_equal(output[1], expected[1])) return false;

    return true;
}

// =============================================================================
// Test 2: Edge Case — compiling an empty network must fail cleanly, not crash
// =============================================================================
bool test_network_empty_compile_fails() {
    EngineContext context;
    Network network; // no layers added

    EngineStatus status = network.compile(context, {2});

    // We EXPECT failure here — an empty network is not a valid, runnable model.
    return !status.is_ok();
}

// =============================================================================
// Test 3: Edge Case — forward() before compile() must fail cleanly
// =============================================================================
bool test_network_forward_without_compile_fails() {
    EngineContext context;
    Network network;

    Tensor weights({2, 2});
    Tensor bias({2});
    network.add_layer(std::make_unique<DenseLayer>(
        "test_dense", 2, 2, std::move(weights), std::move(bias)
    ));
    // Deliberately skipping network.compile(context, {2}) here

    Tensor input({2});
    Tensor output({2});
    EngineStatus status = network.forward(context, input, output);

    return !status.is_ok();
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

    run("test_network_single_dense_layer", test_network_single_dense_layer);
    run("test_network_empty_compile_fails", test_network_empty_compile_fails);
    run("test_network_forward_without_compile_fails", test_network_forward_without_compile_fails);

    std::cout << "\n" << passed << "/" << total << " tests passed.\n";
    return (passed == total) ? 0 : 1;
}