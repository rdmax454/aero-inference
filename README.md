# AeroInference

AeroInference is a lightweight TinyML inference engine written in C++17, built for running small neural network models (Conv2D, Fully Connected, ReLU) on resource-constrained systems. It includes a desktop developer dashboard (Dear ImGui) for loading models, running inference, and viewing latency telemetry.

**Project status: Tested prototype (alpha).** The core math, model parsing, and GUI pipeline are implemented, unit-tested, and verified against a real trained neural network. It has not been validated on embedded hardware. See [Implementation Status](#implementation-status) for what is done and what is planned.

## Verified With a Real Trained Model

A small CNN (Conv2D, ReLU, Conv2D, ReLU, FullyConnected) was trained in PyTorch on the MNIST handwritten digit dataset and reached 96.25% test accuracy in PyTorch. The trained weights were exported to this engine's JSON format and run through the C++ inference path. On 100 randomly sampled, unseen MNIST test images the C++ engine scored 94%, close to the PyTorch result. The gap is consistent with the small sample size.

Performance (desktop, 7th-gen Intel i7, MinGW g++ 14.2): the MNIST CNN averages about 40–50 µs per forward pass with -O2, versus about 240–255 µs without optimisation (1,000-run average, same numerical output). Embedded timings are not yet measured.

This verifies that the Conv2D, ReLU, and FullyConnected kernels give correct results on real trained weights, not just that they run without crashing.

## What Works Today

- **Tensor**: flat, contiguous N-dimensional float buffer with stride-based indexing and `reshape_view()`, which lets pre-allocated buffers be reused across layers with different output sizes. Move-only and zero-initialized.
- **MathBackend**: fully connected forward pass, in-place ReLU, and 2D convolution with stride and padding.
- **Network**: layer graph with shape validation (`compile()`) and ping-pong buffer execution (`forward()`), so inference does not allocate per call.
- **JsonParser**: hand-written scanner with no external JSON dependency. It builds `DenseLayer`, `Conv2DLayer`, and `ActivationLayer` instances from a model file and returns an error status on malformed numeric fields instead of crashing.
- **GUI** (Dear ImGui, GLFW, OpenGL 3.3): loads a model through the real parser, runs a real forward pass, and plots measured latency using `std::chrono`. The dashboard loads the small demo model, `models/drone_nav_v4.json`. The MNIST model is exercised by the test programs.
- **Tests**: 18 unit and integration tests (math kernels, Tensor, Network, Quantizer) plus two MNIST tests and a benchmark (tests/bench_mnist.cpp). See `tests/`.
- **CI**: builds and runs all test programs on Ubuntu and Windows, runs them under ASan/UBSan, and cross-compiles the core for ARM Cortex-M4 on every push.

## What Is Not Done

- No SIMD (AVX/NEON) code. Any vectorization comes from the compiler and is not verified.
- No embedded run yet. The core compiles for ARM Cortex-M4 in CI, but it has not been run on hardware or an emulator
- Quantization is simulated (fake-quantization) in FP32. There are no real INT8 kernels yet
- Supported layers are Conv2D, ReLU, and FullyConnected only. There is no pooling, softmax, batch normalization, recurrent, or attention layer, so the engine handles fixed-size classification and not object detection or sequence models.
- `models/drone_nav_v4.json` has random untrained weights and exists only as a structural test fixture.
- Single developer, no external code review.

## Repository Structure

```text
aero-inference/
├── CMakeLists.txt
├── README.md
├── .github/workflows/       # CI: build and test on every push
├── core/
│   ├── include/             # EngineContext, Tensor, MathBackend, Network, layers/
│   └── src/                 # Network.cpp, MathBackend.cpp
├── compiler/
│   ├── include/             # ModelParser, JsonParser, Quantizer
│   └── src/                 # JsonParser.cpp, Quantizer.cpp
├── gui/
│   ├── include/             # DashboardApp, MainWindow
│   └── src/                 # main.cpp, DashboardApp.cpp, MainWindow.cpp
├── third_party/
│   └── imgui/               # Dear ImGui and GLFW/OpenGL3 backends
├── models/
│   ├── drone_nav_v4.json    # Small untrained fixture for the GUI
│   ├── mnist_model.json     # PyTorch-trained MNIST weights
│   ├── sample_digit.json    # One MNIST test image and its label
│   └── mnist_batch.json     # 100 MNIST test images and labels
├── tests/
│   ├── test_math.cpp
│   ├── test_tensor.cpp
│   ├── test_network.cpp
│   ├── test_mnist_inference.cpp
│   ├── test_mnist_batch_accuracy.cpp
|   ├── bench_mnist.cpp
|   └── test_quantizer.cpp
├── docs/
│   └── SRS.md               # Software Requirements Specification
└── build/                   # Build output (gitignored)
```

## Building

Requires MinGW-w64 g++ (C++17) and GLFW (headers and `libglfw3.a`) available to the compiler. See `.vscode/tasks.json` for the exact build command. Build with `Ctrl+Shift+B` in VS Code, then run `.\build\AeroInference.exe` from the project root, because model paths are relative.

## Running Tests

Each test builds and runs independently. For example:

```powershell
g++ -std=gnu++17 -I core/include -I compiler/include tests/test_mnist_batch_accuracy.cpp core/src/Network.cpp core/src/MathBackend.cpp compiler/src/JsonParser.cpp -o build/test_mnist_batch_accuracy.exe
.\build\test_mnist_batch_accuracy.exe
```

## Implementation Status

| Component | Status |
|---|---|
| Tensor | Implemented |
| MathBackend (FC, ReLU, Conv2D) | Implemented, verified on a trained model |
| Network compile/forward | Implemented, buffer-reuse bug fixed and tested |
| JsonParser | Implemented, numeric-field crash risk fixed |
| Quantizer | Implemented (fake-quantization simulation), tested”.
   Automated tests: “Implemented (18 unit/integration tests, 2 MNIST tests) |
| GUI load/run | Implemented |
| Automated tests | Implemented (12 unit/integration tests, 2 MNIST tests) |
| Trained model verification | Done (MNIST, 94% on 100 samples) |
| CI |All test programs on Ubuntu and Windows, plus sanitizers and an ARM build check |
| SIMD optimization | Not implemented |
| Embedded target | Compiles for Cortex-M4; not run on hardware |
