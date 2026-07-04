#pragma once

#include "ModelParser.hpp"
#include <string>
#include <memory>

/**
 * @brief Zero-Header-Pollution JSON Model Parser and Structure Compiler.
 * Implements the ModelParser contract to translate structural layout strings into network graphs.
 */
class JsonParser final : public ModelParser {
private:
    // Forward declaration of internal engine state parsing options to hide heavy tracking variables
    struct Impl;
    std::unique_ptr<Impl> m_impl;

public:
    /**
     * @brief Constructor allocating internal parser memory buffers safely.
     */
    JsonParser() noexcept;
    
    /**
     * @brief Custom Destructor cleanly handling the unique_ptr forward declaration requirements.
     */
    ~JsonParser() override;

    // Enforce strict asset ownership contracts: ban accidental duplication paths
    JsonParser(const JsonParser&) = delete;
    JsonParser& operator=(const JsonParser&) = delete;
    
    JsonParser(JsonParser&&) noexcept = default;
    JsonParser& operator=(JsonParser&&) noexcept = default;

    /**
     * @brief Parses an offline human-readable JSON structural model from disk.
     * @param context Global runtime engine safety manager.
     * @param file_path System path pointing to the serialized .json network file.
     * @param network Target runtime network instance to populate with layer data.
     */
    [[nodiscard]] EngineStatus parse_from_file(
        EngineContext& context,
        std::string_view file_path,
        Network& network
    ) noexcept override;

    /**
     * @brief Overloaded extraction module pulling JSON strings from raw volatile memory buffers.
     * @param context Global runtime engine safety manager.
     * @param buffer Pointer to the raw ASCII string sequence representing the network parameters.
     * @param buffer_size The total byte size of the binary input string.
     * @param network Target runtime network instance to populate.
     */
    [[nodiscard]] EngineStatus parse_from_buffer(
        EngineContext& context,
        const uint8_t* buffer,
        size_t buffer_size,
        Network& network
    ) noexcept override;
};