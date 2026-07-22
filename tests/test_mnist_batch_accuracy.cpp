#include "Network.hpp"
#include "JsonParser.hpp"
#include "EngineContext.hpp"
#include "Tensor.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>

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
        try { values.push_back(std::stof(token)); } catch (...) { continue; }
    }
    return values;
}

int main() {
    EngineContext context;
    Network network;
    JsonParser parser;

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

    std::ifstream file("models/mnist_batch.json");
    if (!file.is_open()) {
        std::cout << "[FAIL] Could not open mnist_batch.json\n";
        return 1;
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    int count = std::stoi(extract_field(content, "count"));
    std::vector<float> labels_raw = parse_float_array(extract_field(content, "labels"));
    std::vector<float> all_pixels = parse_float_array(extract_field(content, "pixels"));

    if (static_cast<int>(labels_raw.size()) != count || static_cast<int>(all_pixels.size()) != count * 784) {
        std::cout << "[FAIL] Batch file size mismatch. Expected " << count
                   << " labels and " << (count * 784) << " pixels, got "
                   << labels_raw.size() << " labels and " << all_pixels.size() << " pixels.\n";
        return 1;
    }

    int correct = 0;
    Tensor output({10});

    for (int sample = 0; sample < count; ++sample) {
        Tensor input({1, 28, 28});
        for (int p = 0; p < 784; ++p) {
            input[p] = all_pixels[sample * 784 + p];
        }

        EngineStatus forward_status = network.forward(context, input, output);
        if (!forward_status.is_ok()) {
            std::cout << "[FAIL] Forward pass failed on sample " << sample
                       << ": " << context.get_last_error() << "\n";
            continue;
        }

        int predicted = 0;
        float highest = output[0];
        for (size_t i = 1; i < output.size(); ++i) {
            if (output[i] > highest) { highest = output[i]; predicted = static_cast<int>(i); }
        }

        int true_label = static_cast<int>(labels_raw[sample]);
        if (predicted == true_label) correct++;
    }

    std::cout << "\nC++ engine accuracy on " << count << " real test samples: "
               << (100.0 * correct / count) << "%\n";
    std::cout << "(" << correct << " / " << count << " correct)\n";

    return 0;
}