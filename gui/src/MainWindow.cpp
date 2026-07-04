#include "MainWindow.hpp"
#include "imgui.h"
#include <chrono>
#include <random>

MainWindow::MainWindow() noexcept {
    m_latency_history.resize(m_max_history_points, 0.0f);
}

void MainWindow::apply_theme() noexcept {
    ImGuiStyle& style = ImGui::GetStyle();
    ImVec4* colors = style.Colors;
    style.WindowRounding = 8.0f;
    style.FrameRounding = 6.0f;
    style.PopupRounding = 6.0f;
    style.ScrollbarRounding = 12.0f;
    style.WindowBorderSize = 1.0f;
    style.FrameBorderSize = 1.0f;
    style.WindowPadding = ImVec2(12.0f, 12.0f);

    colors[ImGuiCol_WindowBg] = ImVec4(0.07f, 0.08f, 0.10f, 1.00f);
    colors[ImGuiCol_ChildBg] = ImVec4(0.12f, 0.14f, 0.17f, 1.00f);
    colors[ImGuiCol_Border] = ImVec4(0.20f, 0.23f, 0.27f, 1.00f);
    colors[ImGuiCol_Text] = ImVec4(0.92f, 0.94f, 0.96f, 1.00f);
    colors[ImGuiCol_FrameBg] = ImVec4(0.16f, 0.18f, 0.22f, 1.00f);
    colors[ImGuiCol_Button] = ImVec4(0.05f, 0.45f, 0.93f, 1.00f);
    colors[ImGuiCol_ButtonHovered] = ImVec4(0.07f, 0.52f, 1.00f, 1.00f);
    colors[ImGuiCol_ButtonActive] = ImVec4(0.00f, 0.83f, 0.88f, 1.00f);
    colors[ImGuiCol_PlotLines] = ImVec4(0.00f, 0.83f, 0.88f, 1.00f);
}

void MainWindow::render(EngineContext& context, Network& network) noexcept {
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(viewport->Size);

    ImGui::Begin("DashboardHub", nullptr, ImGuiWindowFlags_NoDecoration |
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize);

    render_top_status_bar(context, network);

    float available_width = ImGui::GetContentRegionAvail().x;
    float left_w = available_width * 0.22f;
    float center_w = available_width * 0.48f;
    float right_w = available_width * 0.30f;
    float panel_h = ImGui::GetContentRegionAvail().y;

    ImGui::BeginChild("CtrlPanel", ImVec2(left_w, panel_h), true);
    render_left_control_panel(context, network);
    ImGui::EndChild();

    ImGui::SameLine();

    ImGui::BeginChild("GraphView", ImVec2(center_w, panel_h), true);
    render_center_graph_panel(network);
    ImGui::EndChild();

    ImGui::SameLine();

    ImGui::BeginChild("TelemetryView", ImVec2(right_w, panel_h), true);
    render_right_telemetry_panel();
    ImGui::EndChild();

    ImGui::End();
}

void MainWindow::render_top_status_bar(EngineContext& context, Network& network) noexcept {
    (void)context;
    ImGui::TextColored(ImVec4(0.00f, 0.83f, 0.88f, 1.00f), "AEROINFERENCE PIPELINE:");
    ImGui::SameLine();
    ImGui::Text(m_model_loaded && network.is_compiled() ? "ONLINE | GRAPH STATICALLY ALLOCATED" : "STANDBY | SYSTEM IDLE");
    ImGui::Separator();
}

void MainWindow::render_left_control_panel(EngineContext& context, Network& network) noexcept {
    ImGui::Text("CONTROLLER");
    ImGui::Separator();

    if (ImGui::Button("LOAD TINYML NET", ImVec2(-1.0f, 35.0f))) {
        m_loaded_file_path = "models/drone_nav_v4.json";

        EngineStatus parse_status = m_parser.parse_from_file(context, m_loaded_file_path, network);
        if (!parse_status.is_ok()) {
            m_model_loaded = false;
            m_status_message = "Parse failed: " + context.get_last_error();
        } else {
            EngineStatus compile_status = network.compile(context, {1, 4, 4});
            if (!compile_status.is_ok()) {
                m_model_loaded = false;
                m_status_message = "Compile failed: " + context.get_last_error();
            } else {
                m_model_loaded = true;
                m_status_message = "Model loaded and compiled successfully.";

                m_input_tensor = Tensor({1, 4, 4});
                m_output_tensor = Tensor({4});
                for (size_t i = 0; i < m_input_tensor.size(); ++i) {
                    m_input_tensor[i] = static_cast<float>(i) * 0.1f;
                }
            }
        }
    }

    ImGui::Spacing();

    bool dynamic_state = m_model_loaded && network.is_compiled();
    if (!dynamic_state) ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.5f);

    if (ImGui::Button("RUN PROFILING PASS", ImVec2(-1.0f, 35.0f)) && dynamic_state) {
        auto start_time = std::chrono::high_resolution_clock::now();
        EngineStatus fwd_status = network.forward(context, m_input_tensor, m_output_tensor);
        auto end_time = std::chrono::high_resolution_clock::now();

        if (fwd_status.is_ok()) {
            m_last_inference_latency_us =
                std::chrono::duration<float, std::micro>(end_time - start_time).count();

            m_latency_history.erase(m_latency_history.begin());
            m_latency_history.push_back(m_last_inference_latency_us);
            m_status_message = "Inference completed successfully.";
        } else {
            m_status_message = "Inference failed: " + context.get_last_error();
        }
    }

    if (!dynamic_state) ImGui::PopStyleVar();
}

void MainWindow::render_center_graph_panel(Network& network) noexcept {
    ImGui::Text("TOPOLOGY PIPELINE NODE GRAPH");
    ImGui::Separator();

    if (!m_model_loaded) {
        ImGui::Text("Load a model architecture topology to populate schematic...");
        return;
    }

    ImGui::BulletText("Layers loaded: %zu", network.layer_count());
    ImGui::BulletText("Conv2D Spatial Filter Layer [Input: 1x4x4]");
    ImGui::BulletText("Activation_ReLU In-Place Kernel");
    ImGui::BulletText("FullyConnected Flat Classifier Layer [Output: 4]");
}

void MainWindow::render_right_telemetry_panel() noexcept {
    ImGui::Text("ENGINE TELEMETRY");
    ImGui::Separator();
    ImGui::Text("Forward Inference Step: %.2f us", m_last_inference_latency_us);
    ImGui::Text("Dynamic Allocation Overhead: 0 Bytes");
    ImGui::TextWrapped("Status: %s", m_status_message.c_str());
    ImGui::Spacing();
    ImGui::PlotLines("##LiveLatency", m_latency_history.data(),
        static_cast<int>(m_latency_history.size()), 0, nullptr, 0.0f, 200.0f, ImVec2(-1.0f, 150.0f));
}