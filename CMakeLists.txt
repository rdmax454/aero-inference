cmake_minimum_required(VERSION 3.14)
project(AeroInference CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------------------------------------------------------
# 1. Gather Subsystem Source Files
# -----------------------------------------------------------------------------
file(GLOB_RECURSE CORE_SOURCES "core/src/*.cpp")
file(GLOB_RECURSE COMPILER_SOURCES "compiler/src/*.cpp")

# Explicitly gather your third-party ImGui files
set(IMGUI_SOURCES
    gui/third_party/imgui/imgui.cpp
    gui/third_party/imgui/imgui_draw.cpp
    gui/third_party/imgui/imgui_widgets.cpp
    gui/third_party/imgui/imgui_tables.cpp
    gui/third_party/imgui/backends/imgui_impl_glfw.cpp
    gui/third_party/imgui/backends/imgui_impl_opengl3.cpp
)

set(GUI_SOURCES
    gui/src/main.cpp
    gui/src/DashboardApp.cpp
    gui/src/MainWindow.cpp
)

# -----------------------------------------------------------------------------
# 2. Define Executable Target
# -----------------------------------------------------------------------------
add_executable(AeroInferenceWks
    ${CORE_SOURCES}
    ${COMPILER_SOURCES}
    ${IMGUI_SOURCES}
    ${GUI_SOURCES}
)

# -----------------------------------------------------------------------------
# 3. Global Include Directories Mapping (Fixes the Compiler Paths)
# -----------------------------------------------------------------------------
target_include_directories(AeroInferenceWks PRIVATE
    core/include
    compiler/include
    gui/include
    gui/third_party/imgui
    gui/third_party/imgui/backends
)

# -----------------------------------------------------------------------------
# 4. Link System Dependencies (GLFW and OpenGL)
# -----------------------------------------------------------------------------
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)

target_link_libraries(AeroInferenceWks PRIVATE
    glfw
    OpenGL::GL
)