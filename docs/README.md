# AeroInference

AeroInference is a lightweight TinyML inference engine written in C++17/20, built for running small neural network models (Conv2D, Fully Connected, ReLU) on resource-constrained systems. It includes a desktop developer dashboard (Dear ImGui) for loading models, running inference, and viewing latency telemetry.

**Project status: Working prototype (pre-alpha).** Core math, model parsing, and the GUI pipeline are functional and verified end-to-end with a small test model. It is not yet hardened, tested at scale, or validated on embedded hardware. See [Implementation Status](#implementation-status) below for an honest breakdown of what is done vs. planned.

---

## What Actually Works Today

- **Tensor** — flat, contiguous N-dimensional float buffer with stride-based indexing. Move-only (no copy), zero-initialized on allocation.
- **MathBackend** — correct, tested-by-inspection implementations of:
  - Fully connected (dense) forward pass
  - ReLU activation (in-place)
  - 2D convolution with stride and padding support
- **Network** — layer graph container with shape validation (`compile()`) and ping-pong buffer execution (`forward()`) to avoid per-inference heap allocation.
- **JsonParser** — hand-written JSON-like scanner (no external JSON library dependency) that reads a model file from disk and builds real `DenseLayer` / `Conv2DLayer` / `ActivationLayer` instances with real weight data. Includes safe numeric parsing (malformed files return an error instead of crashing).
- **GUI (Dear ImGui + GLFW + OpenGL 3.3)** — desktop dashboard with:
  - "Load Model" button that calls the real parser and reports real success/failure
  - "Run Inference" button that runs a real forward pass and measures real elapsed time via `std::chrono`
  - Live latency history plot

## What Is NOT Yet True (Despite Earlier Drafts Implying It)

- No SIMD (AVX/NEON) is implemented. Math relies on the compiler's automatic vectorization, which is not guaranteed or verified.
- No automated test suite exists yet (`tests/` is currently empty).
- No embedded/cross-compilation target has been attempted. This currently runs only as a desktop Windows application.
- `Quantizer` (FP32→INT8) has a defined interface but its implementation has not yet been reviewed or verified.
- The demo model shipped with the project (`models/drone_nav_v4.json`) has random, untrained weights — it proves the pipeline works, not that the network produces meaningful predictions.
- No CI/CD, no code review process, single-developer project to date.

---

## Repository Structure (matches actual disk layout)

```text
aero-inference/
├── CMakeLists.txt
├── README.md
├── core/
│   ├── include/            # EngineContext, Tensor, MathBackend, Network, layers/
│   └── src/                # Network.cpp, MathBackend.cpp
├── compiler/
│   ├── include/            # ModelParser, JsonParser, Quantizer
│   └── src/                # JsonParser.cpp, Quantizer.cpp
├── gui/
│   ├── include/            # DashboardApp, MainWindow
│   └── src/                # main.cpp, DashboardApp.cpp, MainWindow.cpp
├── third_party/
│   └── imgui/               # Dear ImGui + backends (GLFW/OpenGL3)
├── models/
│   └── drone_nav_v4.json    # Small demo model (untrained, structural test only)
├── tests/                   # Currently empty — planned
└── build/                   # Build output (gitignored)
```

---

## Building

Requires MinGW-w64 g++ (C++17) and GLFW (headers + `libglfw3.a`) available to the compiler. See `.vscode/tasks.json` for the exact build command. Run via `Ctrl+Shift+B` in VS Code, or invoke the same `g++` command directly from a terminal.

## Implementation Status

| Component | Status |
|---|---|
| Tensor | Implemented |
| MathBackend (FC, ReLU, Conv2D) | Implemented |
| Network compile/forward | Implemented |
| JsonParser | Implemented, crash-hardened |
| Quantizer | Interface defined, implementation unverified |
| GUI load/run wiring | Implemented |
| Automated tests | Not started |
| SIMD optimization | Not implemented |
| Embedded target | Not attempted |
| CI/CD | Not set up |
