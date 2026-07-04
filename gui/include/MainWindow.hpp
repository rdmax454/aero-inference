#pragma once

#include "../../core/include/Network.hpp"
#include "../../core/include/EngineContext.hpp"
#include "../../core/include/Tensor.hpp"
#include "../../compiler/include/JsonParser.hpp"
#include <string>
#include <vector>

/**
 * @brief Handles rendering layout panels, styling, and charts.
 * Pure immediate-mode layout engine isolated from window management.
 */
class MainWindow {
private:
    float m_last_inference_latency_us{0.0f};
    std::vector<float> m_latency_history;
    const size_t m_max_history_points{100};
    bool m_model_loaded{false};
    std::string m_loaded_file_path{"None"};

    JsonParser m_parser;
    Tensor m_input_tensor;
    Tensor m_output_tensor;
    std::string m_status_message{"No model loaded yet."};

    void render_top_status_bar(EngineContext& context, Network& network) noexcept;
    void render_left_control_panel(EngineContext& context, Network& network) noexcept;
    void render_center_graph_panel(Network& network) noexcept;
    void render_right_telemetry_panel() noexcept;

public:
    MainWindow() noexcept;
    ~MainWindow() = default;

    MainWindow(const MainWindow&) = delete;
    MainWindow& operator=(const MainWindow&) = delete;
    MainWindow(MainWindow&&) noexcept = default;
    MainWindow& operator=(MainWindow&&) noexcept = default;

    void apply_theme() noexcept;
    void render(EngineContext& context, Network& network) noexcept;
};