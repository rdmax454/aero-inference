#pragma once

#include <string>
#include <string_view>
#include <iostream>
#include <cstdint>

/**
 * @brief Deterministic status codes for the inference runtime pipeline.
 * Replaces high-overhead C++ exception throwing with embedded-safe enumeration results.
 */
enum class StatusCode : uint8_t {
    Success = 0,
    Error_NullPointer,
    Error_DimensionMismatch,
    Error_MemoryAllocationFailed,
    Error_LayerInitializationFailed,
    Error_InvalidModelFile,
    Error_QuantizationOverflow,
    Error_ExecutionFailure
};

/**
 * @brief Enterprise status wrapper tracking pipeline states and runtime telemetry metrics.
 */
struct EngineStatus {
    StatusCode code{StatusCode::Success};
    std::string_view message{"Operation completed successfully."};

    [[nodiscard]] constexpr bool is_ok() const noexcept { 
        return code == StatusCode::Success; 
    }
};

/**
 * @brief Globally accessible Context state configuration manager.
 * Handles operational modes, tracking telemetry data metrics without leaking state.
 */
class EngineContext {
private:
    bool m_debug_mode{true};
    std::string m_last_error_log{"None"};

public:
    constexpr EngineContext() noexcept = default;
    ~EngineContext() = default;

    // Delete copy operations to ensure strict tracking context identity across execution domains
    EngineContext(const EngineContext&) = delete;
    EngineContext& operator=(const EngineContext&) = delete;
    
    EngineContext(EngineContext&&) noexcept = default;
    EngineContext& operator=(EngineContext&&) noexcept = default;

    /**
     * @brief Formats, isolates, and logs error pipeline telemetry messages smoothly.
     * @param code The system status evaluation identifier.
     * @param system_message Contextual breakdown of where the pipeline halted.
     */
    void log_error(StatusCode code, std::string_view system_message) noexcept {
        m_last_error_log = std::string(system_message);
        if (m_debug_mode) {
            std::cerr << "[AeroInference Runtime Alert | Code " 
                      << static_cast<int>(code) << "]: " 
                      << system_message << "\n";
        }
    }

    [[nodiscard]] std::string get_last_error() const { 
        return m_last_error_log; 
    }
    
    void set_debug_mode(bool enable) noexcept { 
        m_debug_mode = enable; 
    }
    
    [[nodiscard]] bool is_debug_active() const noexcept { 
        return m_debug_mode; 
    }
};