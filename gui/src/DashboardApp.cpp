#include "DashboardApp.hpp"
#include <iostream>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "GLFW/glfw3.h"

DashboardApp::~DashboardApp() {
    shutdown();
}

bool DashboardApp::initialize(int width, int height, const char* window_title) noexcept {
    if (!glfwInit()) {
        std::cerr << "[Fatal Error]: Failed to initialize system GLFW drivers.\n";
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    m_window_handle = glfwCreateWindow(width, height, window_title, nullptr, nullptr);
    if (!m_window_handle) {
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(m_window_handle);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    m_ui_window.apply_theme();

    ImGui_ImplGlfw_InitForOpenGL(m_window_handle, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    m_is_initialized = true;
    return true;
}

void DashboardApp::run_loop() noexcept {
    if (!m_is_initialized) return;

    while (!glfwWindowShouldClose(m_window_handle)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        m_ui_window.render(m_context, m_network);

        ImGui::Render();

        int display_w, display_h;
        glfwGetFramebufferSize(m_window_handle, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.07f, 0.08f, 0.10f, 1.00f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(m_window_handle);
    }
}

void DashboardApp::shutdown() noexcept {
    if (m_is_initialized) {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        glfwDestroyWindow(m_window_handle);
        glfwTerminate();
        m_is_initialized = false;
    }
}