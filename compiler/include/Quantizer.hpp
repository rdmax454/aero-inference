#pragma once

#include "Tensor.hpp"
#include "EngineContext.hpp"
#include <vector>
#include <cstdint>

/**
 * @brief Industrial Post-Training Quantization Optimization Engine.
 * Scales weights from FP32 down to INT8 configurations to reduce memory footprint.
 */
class Quantizer {
private:
    float m_scale{1.0f};
    int8_t m_zero_point{0};
    bool m_is_calibrated{false};

public:
    constexpr Quantizer() noexcept = default;
    ~Quantizer() = default;

    // Enforce strict asset constraints: prevent duplicate scaling utility states
    Quantizer(const Quantizer&) = delete;
    Quantizer& operator=(const Quantizer&) = delete;
    Quantizer(Quantizer&&) noexcept = default;
    Quantizer& operator=(Quantizer&&) noexcept = default;

    /**
     * @brief Analyzes a reference tensor to compute absolute dynamic scalar mapping ranges.
     * Math Logic: Finds max(|min|, |max|) to build a symmetric quantization scale factor.
     * @param context Global runtime engine safety manager.
     * @param reference_tensor The original raw FP32 target tensor to analyze.
     * @return EngineStatus Status tracking indicating calibration validity.
     */
    [[nodiscard]] EngineStatus calibrate_symmetric(
        EngineContext& context,
        const Tensor& reference_tensor
    ) noexcept;

    /**
     * @brief Transforms a high-precision FP32 tensor into compressed INT8 elements, 
     * then scales it back to a flat FP32 tensor to simulate fake quantization performance.
     * Math execution formula: Quantized = clamp(round(Value / Scale) + ZeroPoint)
     * @param context Global runtime engine safety manager.
     * @param input Source floating-point tensor to compress.
     * @param output Target destination tensor holding optimized, scaled representations.
     * @return EngineStatus Code validation verification metrics.
     */
    [[nodiscard]] EngineStatus quantize_tensor(
        EngineContext& context,
        const Tensor& input,
        Tensor& output
    ) const noexcept;

    // Quantization Parameters Getters
    [[nodiscard]] constexpr float scale() const noexcept { return m_scale; }
    [[nodiscard]] constexpr int8_t zero_point() const noexcept { return m_zero_point; }
    [[nodiscard]] constexpr bool is_calibrated() const noexcept { return m_is_calibrated; }

    /**
     * @brief Explicitly forces custom mapping scales if parameters are loaded from metadata.
     */
    void force_parameters(float custom_scale, int8_t custom_zero_point) noexcept {
        m_scale = custom_scale;
        m_zero_point = custom_zero_point;
        m_is_calibrated = true;
    }
};