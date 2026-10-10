# Software Requirements Specification
## AeroInference: TinyML Edge Inference Engine

**Document Version:** 1.1
**Status:** Draft (retrofitted from an existing prototype, see Section 1.5)
**Author:** M. Hassaan

---

## 1. Introduction

### 1.1 Purpose
This document specifies the functional and non-functional requirements for AeroInference, a lightweight neural network inference engine intended for resource-constrained (embedded/edge) environments, together with its desktop developer dashboard.

### 1.2 Scope
AeroInference shall:
- Parse a serialized neural network model definition from a JSON file.
- Execute forward-pass inference (Convolution, Fully Connected, ReLU layers) on that model.
- Report execution status and latency telemetry through a desktop GUI.
- (Planned) Run on embedded/microcontroller targets and support INT8 quantized models.

### 1.3 Definitions, Acronyms, Abbreviations
| Term | Meaning |
|---|---|
| Tensor | Flat, contiguous, N-dimensional numeric buffer |
| Forward pass | Running input data through a network to produce output, without training |
| Quantization | Reducing numeric precision (e.g., FP32 to INT8) to save memory and compute |
| TinyML | Machine learning models designed to run on constrained hardware |
| SRS | Software Requirements Specification (this document) |

### 1.4 References
- Source code: `core/`, `compiler/`, `gui/`, `tests/` (see repository)
- Dear ImGui documentation (third-party GUI library in use)

### 1.5 Overview and Origin Note
Under normal practice an SRS is written before implementation. In this project a working prototype existed first, and this SRS was written afterward to capture what the system does and should do. It is stated here so that anyone maintaining the project knows the document's origin.

---

## 2. Overall Description

### 2.1 Product Perspective
AeroInference is a standalone desktop application (Windows, currently) made of three decoupled layers: a core math and execution engine, a model parser layer, and a GUI dashboard. It has no network dependencies and does not integrate with any cloud service, database, or third-party API beyond the GLFW and OpenGL windowing libraries.

### 2.2 Product Functions (Summary)
1. Load a neural network model definition from a JSON file.
2. Validate the model's structural integrity (layer shapes and dimension compatibility).
3. Execute a forward-pass inference on an input tensor.
4. Display latency and status telemetry to the user.
5. (Planned) Quantize model weights from FP32 to INT8.
6. (Planned) Run on an embedded microcontroller target.

### 2.3 User Characteristics
Primary user: a developer or engineer integrating or testing a small classification model, comfortable with technical dashboards and JSON model files.

### 2.4 Constraints
- Must avoid heap allocation during the inference forward pass.
- Must not let C++ exceptions cross `noexcept` boundaries (see FR-9).
- Supported layers are Conv2D, ReLU, and FullyConnected only.
- Current toolchain is MinGW-w64 g++ on Windows. No cross-platform build has been verified.

### 2.5 Assumptions and Dependencies
- GLFW and OpenGL 3.3 are available on the host system.
- Model JSON files are well-formed enough for the custom lightweight parser. It is not a full RFC 8259 JSON parser.

---

## 3. Specific Requirements

### 3.1 Functional Requirements

| ID | Requirement | Status |
|---|---|---|
| FR-1 | The system shall load a model definition from a specified JSON file path. | Implemented |
| FR-2 | The system shall construct in-memory layer objects (Conv2D, FullyConnected, Activation_ReLU) from the parsed file, including weight and bias data. | Implemented |
| FR-3 | The system shall return a descriptive error, not crash, when the model file is missing, malformed, or has mismatched dimensions. | Implemented for dimension fields; not fuzz-tested |
| FR-4 | The system shall validate that consecutive layer shapes are compatible before allowing inference (`compile()`). | Implemented |
| FR-5 | The system shall execute a forward pass through all loaded layers given an input tensor. | Implemented, including layers with differing output sizes |
| FR-6 | The system shall measure and display the wall-clock time of a forward pass in microseconds. | Implemented |
| FR-7 | The system shall provide a GUI control to load a model and display success or failure. | Implemented |
| FR-8 | The system shall provide a GUI control to run a forward pass and display the latency. | Implemented |
| FR-9 | Core execution-path functions shall not throw uncaught C++ exceptions. | Partially implemented: JSON numeric parsing hardened; other modules not yet audited |
| FR-10 | The system shall support post-training quantization of model weights from FP32 to INT8. | Interface defined only; logic not implemented or verified |
| FR-11 | The system shall support execution on an embedded microcontroller target. | Not started |
| FR-12 | The system shall provide an automated test suite validating math kernel correctness against known reference values. | Implemented: 12 unit and integration tests plus 2 MNIST tests |
| FR-13 | The engine shall reproduce the predictions of a trained reference model (PyTorch) within a stated accuracy tolerance. | Implemented: 94% on 100 unseen MNIST samples vs 96.25% in PyTorch |

### 3.2 Non-Functional Requirements

| ID | Category | Requirement | Status |
|---|---|---|---|
| NFR-1 | Performance | The forward pass shall execute without heap allocation once the network is compiled. | Implemented (ping-pong buffer design) |
| NFR-2 | Reliability | The application shall not terminate on malformed input files. | Partially met: one known crash class fixed; full audit pending |
| NFR-3 | Portability | The core engine (`core/`, `compiler/`) shall have no dependency on the GUI layer. | Implemented (module separation) |
| NFR-4 | Maintainability | The codebase shall use consistent, documented interfaces (`ILayer`, `ModelParser`). | Implemented |
| NFR-5 | Testability | Critical math functions shall be independently unit-testable. | Implemented: math kernels, Tensor, and Network covered by automated tests |
| NFR-6 | Security | Model files shall not be able to trigger out-of-bounds memory access via malformed dimension data. | Needs verification; not fuzz-tested |

### 3.3 External Interface Requirements
- **File I/O:** reads `.json` model files from a local path.
- **GUI:** desktop window (GLFW, OpenGL 3.3, Dear ImGui) with keyboard and mouse input.
- **Build system:** g++ command-line invocation (see `.vscode/tasks.json`). The repository contains a `CMakeLists.txt`, but it has not been tested.

---

## 4. Open Items for Next Revision
1. Confirm whether `CMakeLists.txt` is functional or aspirational.
2. Define acceptance criteria for FR-11 once a specific embedded target or emulator is chosen.
3. Numeric correctness (previously open for FR-12) is now defined: unit tests compare against hand-calculated values with an absolute tolerance of 1e-4, and the MNIST test compares C++ accuracy against the PyTorch reference. Extend this with element-wise output comparison against PyTorch if tighter validation is needed.
4. Extend CI to run the Tensor and MNIST tests, which the workflow does not yet include.