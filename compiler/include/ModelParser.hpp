#pragma once

#include <string_view>
#include <memory>
#include "Network.hpp"
#include "EngineContext.hpp"

/**
 * @brief Abstract Base Serialization Interface for Neural Network Model Configurations.
 * Defines the contract for parsing weight payloads and mapping target topologies into the core runtime.
 */
class ModelParser {
public:
    ModelParser() noexcept = default;
    virtual ~ModelParser() = default;

    // Enforce strict compiler asset constraints: prohibit accidental duplication
    ModelParser(const ModelParser&) = delete;
    ModelParser& operator=(const ModelParser&) = delete;
    ModelParser(ModelParser&&) noexcept = default;
    ModelParser& operator=(ModelParser&&) noexcept = default;

    /**
     * @brief Parses an explicit offline model file from disk and maps its properties to an active Network object.
     * @param context Global runtime engine safety manager.
     * @param file_path System path pointing to the serialized network representation.
     * @param network Target runtime network instance to populate with layers and parameter weights.
     * @return EngineStatus Tracking structure indicating file reading accuracy and structural alignment.
     */
    [[nodiscard]] virtual EngineStatus parse_from_file(
        EngineContext& context,
        std::string_view file_path,
        Network& network
    ) noexcept = 0;

    /**
     * @brief Overloaded extraction module pulling model structural matrices from raw volatile memory frames.
     * Crucial for secure environments or embedded devices storing raw models directly within a flash memory block.
     * @param context Global runtime engine safety manager.
     * @param buffer Pointer to the raw binary sequence representing the network parameters.
     * @param buffer_size The total byte constraint size of the binary input stream.
     * @param network Target runtime network instance to populate.
     * @return EngineStatus Tracking structure proving buffer ingestion fidelity.
     */
    [[nodiscard]] virtual EngineStatus parse_from_buffer(
        EngineContext& context,
        const uint8_t* buffer,
        size_t buffer_size,
        Network& network
    ) noexcept = 0;
};