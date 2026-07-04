#include "JsonParser.hpp"
#include "layers/DenseLayer.hpp"
#include "layers/Conv2DLayer.hpp"
#include "layers/ActivationLayer.hpp"
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>

// =============================================================================
// Internal Private Parser State Definition (The Pimpl Idiom Implementation)
// =============================================================================
struct JsonParser::Impl {
    /**
     * @brief Minimalist, ultra-fast embedded token scanner to parse structural values 
     * without relying on bloated external standard object dependencies.
     */
    static std::string extract_value(const std::string& json_str, const std::string& key) noexcept {
        size_t key_pos = json_str.find("\"" + key + "\"");
        if (key_pos == std::string::npos) return "";

        size_t colon_pos = json_str.find(":", key_pos);
        if (colon_pos == std::string::npos) return "";

        size_t start_pos = json_str.find_first_not_of(" \t\n\r\"", colon_pos + 1);
        if (start_pos == std::string::npos) return "";

        size_t end_pos;
        if (json_str[start_pos] == '[') {
            // Target value is an array chunk block
            end_pos = json_str.find("]", start_pos);
            if (end_pos != std::string::npos) end_pos += 1;
        } else {
            // Target value is a standard variable string or numerical entry literal
            end_pos = json_str.find_first_of(",}\n\r]", start_pos);
        }

        if (end_pos == std::string::npos) return json_str.substr(start_pos);
        
        std::string result = json_str.substr(start_pos, end_pos - start_pos);
        if (!result.empty() && result.back() == '"') {
            result.pop_back(); // Clean trailing string quotation marks securely
        }
        return result;
    }

    /**
     * @brief Translates an ASCII array string block into a flat floating-point collection.
     */
    static std::vector<float> parse_float_array(const std::string& array_str) noexcept {
        std::vector<float> values;
        size_t start = array_str.find("[");
        size_t end = array_str.find("]");
        if (start == std::string::npos || end == std::string::npos) return values;

        std::string clean_content = array_str.substr(start + 1, end - start - 1);
        std::stringstream ss(clean_content);
        std::string token;
        
        while (std::getline(ss, token, ',')) {
            try {
                values.push_back(std::stof(token));
            } catch (...) {
                // Return gracefully on formatting anomalies to maintain safety guarantees
                continue;
            }
        }
        return values;
    }

    /**
     * @brief Safely converts a string to size_t without throwing on malformed or missing input.
     * Returns false instead of crashing when the value is empty or non-numeric.
     */
    static bool safe_stoull(const std::string& str, size_t& out_value) noexcept {
        if (str.empty()) return false;
        try {
            out_value = std::stoull(str);
            return true;
        } catch (...) {
            return false;
        }
    }
};

// =============================================================================
// Public Interface Implementation Routing Rules
// =============================================================================
JsonParser::JsonParser() noexcept : m_impl(std::make_unique<Impl>()) {}
JsonParser::~JsonParser() = default;

EngineStatus JsonParser::parse_from_file(
    EngineContext& context,
    std::string_view file_path,
    Network& network
) noexcept {
    std::ifstream file((std::string(file_path)));
    if (!file.is_open()) {
        context.log_error(StatusCode::Error_InvalidModelFile, 
            "Failed to open the specified target JSON model configuration structure file from disk.");
        return EngineStatus{StatusCode::Error_InvalidModelFile, "Unable to access file source path."};
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    return parse_from_buffer(context, reinterpret_cast<const uint8_t*>(content.data()), content.size(), network);
}

EngineStatus JsonParser::parse_from_buffer(
    EngineContext& context,
    const uint8_t* buffer,
    size_t buffer_size,
    Network& network
) noexcept {
    if (!buffer || buffer_size == 0) {
        context.log_error(StatusCode::Error_NullPointer, "Volatile stream parse buffer holds an empty pointer state.");
        return EngineStatus{StatusCode::Error_NullPointer, "Parser buffer parameter invalid."};
    }

    std::string json_content(reinterpret_cast<const char*>(buffer), buffer_size);
    network.clear(); // Reset network configuration tracking prior to processing graph layout

    // Iterate across layer configuration block components sequentially
    size_t search_offset = 0;
    while (true) {
        size_t layer_block_start = json_content.find("{", search_offset);
        if (layer_block_start == std::string::npos) break;

        size_t layer_block_end = json_content.find("}", layer_block_start);
        if (layer_block_end == std::string::npos) break;

        std::string layer_chunk = json_content.substr(layer_block_start, layer_block_end - layer_block_start + 1);
        search_offset = layer_block_end + 1;

        std::string layer_type = m_impl->extract_value(layer_chunk, "type");
        std::string layer_name = m_impl->extract_value(layer_chunk, "name");

        if (layer_type.empty()) continue;

        // ---------------------------------------------------------------------
        // Node Generation Pass 1: FullyConnected (Dense) Configurations
        // ---------------------------------------------------------------------
        if (layer_type == "FullyConnected") {
            size_t in_dim = 0, out_dim = 0;
            if (!m_impl->safe_stoull(m_impl->extract_value(layer_chunk, "input_dim"), in_dim) ||
                !m_impl->safe_stoull(m_impl->extract_value(layer_chunk, "output_dim"), out_dim)) {
                context.log_error(StatusCode::Error_InvalidModelFile,
                    "Malformed or missing dimension fields in FullyConnected layer definition.");
                return EngineStatus{StatusCode::Error_InvalidModelFile, "Bad FullyConnected dimensions."};
            }

            std::vector<float> raw_w = m_impl->parse_float_array(m_impl->extract_value(layer_chunk, "weights"));
            std::vector<float> raw_b = m_impl->parse_float_array(m_impl->extract_value(layer_chunk, "bias"));

            if (raw_w.size() != (in_dim * out_dim) || raw_b.size() != out_dim) {
                context.log_error(StatusCode::Error_DimensionMismatch, 
                    "Weight parameter matrix tracking bounds mismatch inside loaded Dense network segment configuration.");
                return EngineStatus{StatusCode::Error_DimensionMismatch, "Parameter array sizing conflict."};
            }

            Tensor weights_tensor({in_dim, out_dim});
            std::copy(raw_w.begin(), raw_w.end(), weights_tensor.data());

            Tensor bias_tensor({out_dim});
            std::copy(raw_b.begin(), raw_b.end(), bias_tensor.data());

            network.add_layer(std::make_unique<DenseLayer>(
                layer_name, in_dim, out_dim, std::move(weights_tensor), std::move(bias_tensor)
            ));
        }
        // ---------------------------------------------------------------------
        // Node Generation Pass 2: Conv2D Spatial Filter Configurations
        // ---------------------------------------------------------------------
        else if (layer_type == "Conv2D") {
            size_t in_c = 0, out_c = 0, k_size = 0, stride = 0, padding = 0;
            bool ok = m_impl->safe_stoull(m_impl->extract_value(layer_chunk, "in_channels"), in_c) &&
                      m_impl->safe_stoull(m_impl->extract_value(layer_chunk, "out_channels"), out_c) &&
                      m_impl->safe_stoull(m_impl->extract_value(layer_chunk, "kernel_size"), k_size) &&
                      m_impl->safe_stoull(m_impl->extract_value(layer_chunk, "stride"), stride) &&
                      m_impl->safe_stoull(m_impl->extract_value(layer_chunk, "padding"), padding);
            if (!ok) {
                context.log_error(StatusCode::Error_InvalidModelFile,
                    "Malformed or missing dimension fields in Conv2D layer definition.");
                return EngineStatus{StatusCode::Error_InvalidModelFile, "Bad Conv2D dimensions."};
            }

            std::vector<float> raw_w = m_impl->parse_float_array(m_impl->extract_value(layer_chunk, "weights"));

            if (raw_w.size() != (out_c * in_c * k_size * k_size)) {
                context.log_error(StatusCode::Error_DimensionMismatch, 
                    "Spatial filter coordinate data layout dimensions fail validation parameters for loaded Conv2D structural element.");
                return EngineStatus{StatusCode::Error_DimensionMismatch, "Conv2D structural data tracking mismatch."};
            }

            Tensor weights_tensor({out_c, in_c, k_size, k_size});
            std::copy(raw_w.begin(), raw_w.end(), weights_tensor.data());

            network.add_layer(std::make_unique<Conv2DLayer>(
                layer_name, in_c, out_c, k_size, stride, padding, std::move(weights_tensor)
            ));
        }
        // ---------------------------------------------------------------------
        // Node Generation Pass 3: Element-Wise Nonlinear Activation Configurations
        // ---------------------------------------------------------------------
        else if (layer_type == "Activation_ReLU") {
            network.add_layer(std::make_unique<ActivationLayer>(layer_name));
        }
    }

    if (network.layer_count() == 0) {
        context.log_error(StatusCode::Error_LayerInitializationFailed, 
            "JSON file structure parsed, but structural evaluation generated an empty model graph chain layout mapping.");
        return EngineStatus{StatusCode::Error_LayerInitializationFailed, "No valid operational segments extracted."};
    }

    return EngineStatus{StatusCode::Success, "JSON configuration structure parsed and imported successfully."};
}