#include "Network.hpp"
#include "JsonParser.hpp"
#include "EngineContext.hpp"
#include "Tensor.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>

// =============================================================================
// Minimal manual reader for the sample_digit.json data file.
// NOTE: This duplicates a small amount of parsing logic from JsonParser::Impl.
// That's acceptable here because JsonParser is specifically built to parse
// MODEL files (layers/weights), not arbitrary data files — using it for this
// would be forcing the wrong tool to do a different job. A real production
// system might unify these under one generic JSON reader; for now, this
// small, honest duplication is a reasonable, clearly-labeled shortcut.
// =============================================================================
std::string extract_field(const std::string& content, const std::string& key) {
    size_t key_pos = content.find("\"" + key + "\"");
    if (key_pos == std::string::npos) return "";

    size_t colon_pos = content.find(":", key_pos);
    size_t start_pos = content.find_first_not_of(" \t\n\r", colon_pos + 1);

    size_t end_pos;
    if (content[start_pos] == '[') {
        end_pos = content.find("]", start_pos) + 1;
    } else {
        end_pos = content.find_first_of(",}\n", start_pos);
    }
    return content.substr(start_pos, end_pos - start_pos);
}

std::vector<float> parse_float_array(const std::string& array_str) {
    std::vector<float> values;
    size_t start = array_str.find("[");
    size_t end = array_str.find("]");
    std::string clean = array_str.substr(start + 1, end - start - 1);

    std::stringstream ss(clean);
    std::string token;
    while (std::getline(ss, token, ',')) {
        try {
            values.push_back(std::stof(token));
        } catch (...) {
            continue;
        }
    }
    return values;
}

int main() {
    EngineContext context;
    Network network;
    JsonParser parser;

    // -------------------------------------------------------------------
    // Step 1: Load the REAL trained model (not the old random demo model)
    // -------------------------------------------------------------------
    EngineStatus parse_status = parser.parse_from_file(context, "models/mnist_model.json", network);
    if (!parse_status.is_ok()) {
        std::cout << "[FAIL] Could not parse model: " << context.get_last_error() << "\n";
        return 1;
    }

    EngineStatus compile_status = network.compile(context, {1, 28, 28});
    if (!compile_status.is_ok()) {
        std::cout << "[FAIL] Could not compile network: " << context.get_last_error() << "\n";
        return 1;
    }

    std::cout << "Model loaded and compiled successfully. Layers: " << network.layer_count() << "\n";

    // -------------------------------------------------------------------
    // Step 2: Load the real MNIST test image + its true label
    // -------------------------------------------------------------------
    std::ifstream file("models/sample_digit.json");
    if (!file.is_open()) {
        std::cout << "[FAIL] Could not open sample_digit.json\n";
        return 1;
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    int true_label = std::stoi(extract_field(content, "true_label"));
    std::vector<float> pixels = parse_float_array(extract_field(content, "pixels"));

    if (pixels.size() != 784) {
        std::cout << "[FAIL] Expected 784 pixels, got " << pixels.size() << "\n";
        return 1;
    }

    Tensor input({1, 28, 28});
    std::copy(pixels.begin(), pixels.end(), input.data());

    // -------------------------------------------------------------------
    // Step 3: Run the REAL forward pass through your engine
    // -------------------------------------------------------------------
    Tensor output({10});
    EngineStatus forward_status = network.forward(context, input, output);
    if (!forward_status.is_ok()) {
        std::cout << "[FAIL] Forward pass failed: " << context.get_last_error() << "\n";
        return 1;
    }

    // -------------------------------------------------------------------
    // Step 4: argmax — find which of the 10 output scores is highest
    // (C++ equivalent of Python's torch.argmax(output, dim=1))
    // -------------------------------------------------------------------
    int predicted_digit = 0;
    float highest_score = output[0];
    for (size_t i = 1; i < output.size(); ++i) {
        if (output[i] > highest_score) {
            highest_score = output[i];
            predicted_digit = static_cast<int>(i);
        }
    }

    // -------------------------------------------------------------------
    // Step 5: Report the result honestly
    // -------------------------------------------------------------------
    std::cout << "\nTrue label:      " << true_label << "\n";
    std::cout << "Predicted digit: " << predicted_digit << "\n";

    if (predicted_digit == true_label) {
        std::cout << "[MATCH] Your C++ engine agrees with the trained Python model.\n";
        return 0;
    } else {
        std::cout << "[MISMATCH] Something is wrong in the pipeline.\n";
        return 1;
    }
}