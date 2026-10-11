#include "Network.hpp"
#include "JsonParser.hpp"
#include "EngineContext.hpp"
#include "Tensor.hpp"
#include <chrono>
#include <iostream>

int main() {
    EngineContext context;
    context.set_debug_mode(false);
    Network network;
    JsonParser parser;

    if (!parser.parse_from_file(context, "models/mnist_model.json", network).is_ok()) {
        std::cout << "[FAIL] Could not parse model: " << context.get_last_error() << "\n";
        return 1;
    }
    if (!network.compile(context, {1, 28, 28}).is_ok()) {
        std::cout << "[FAIL] Could not compile network: " << context.get_last_error() << "\n";
        return 1;
    }

    Tensor input({1, 28, 28});
    Tensor output({10});
    for (size_t i = 0; i < 784; ++i) input[i] = static_cast<float>(i % 7) / 7.0f;

    const int warmup = 50;
    const int runs = 1000;

    for (int i = 0; i < warmup; ++i) network.forward(context, input, output);

    float sink = 0.0f;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < runs; ++i) {
        network.forward(context, input, output);
        sink += output[0];
    }
    auto stop = std::chrono::steady_clock::now();

    double total_us = std::chrono::duration<double, std::micro>(stop - start).count();
    std::cout << "Runs: " << runs << "\n";
    std::cout << "Average per forward pass: " << (total_us / runs) << " us\n";
    std::cout << "(checksum " << sink << ")\n";
    return 0;
}