#pragma once
#include "../../core/include/EngineContext.hpp"
#include "../../core/include/Network.hpp"
#include "MainWindow.hpp"

struct GLFWwindow;

/**
 * @brief Coordinates desktop system resources, graphic framework contexts, and core run loop windows.
 */
class DashboardApp {
private:
    EngineContext m_context;
    Network m_network;
    MainWindow m_ui_window;
    GLFWwindow* m_window_handle{nullptr};
    bool m_is_initialized{false};

public:
    DashboardApp() noexcept = default;
    ~DashboardApp();

    DashboardApp(const DashboardApp&) = delete;
    DashboardApp& operator=(const DashboardApp&) = delete;
    DashboardApp(DashboardApp&&) noexcept = default;
    DashboardApp& operator=(DashboardApp&&) noexcept = default;

    bool initialize(int width, int height, const char* window_title) noexcept;
    void run_loop() noexcept;
    void shutdown() noexcept;
};