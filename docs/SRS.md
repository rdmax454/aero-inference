# Software Requirements Specification
## AeroInference — TinyML Edge Inference Engine

**Document Version:** 1.0
**Status:** Draft (retrofitted from existing prototype — see Section 1.5)
**Author:** M. Hassaan

---

## 1. Introduction

### 1.1 Purpose
This document specifies the functional and non-functional requirements for AeroInference, a lightweight neural network inference engine intended for resource-constrained (embedded/edge) environments, along with its accompanying desktop developer dashboard.

### 1.2 Scope
AeroInference shall:
- Parse a serialized neural network model definition from a JSON file.
- Execute forward-pass inference (Convolution, Fully Connected, ReLU layers) on that model.
- Report execution status and latency telemetry through a desktop GUI.
- (Planned, not yet in scope of current implementation) Run on embedded/microcontroller targets and support INT8 quantized models.

### 1.3 Definitions, Acronyms, Abbreviations
| Term | Meaning |
|---|---|
| Tensor | Flat, contiguous, N-dimensional numeric buffer |
| Forward pass | Running input data through a network to produce output, without training |
| Quantization | Reducing numeric precision (e.g., FP32 → INT8) to save memory/compute |
| TinyML | Machine learning models designed to run on constrained hardware |
| SRS | Software Requirements Specification (this document) |

### 1.4 References
- Existing source code: `core/`, `compiler/`, `gui/` (see repository)
- Dear ImGui documentation (third-party GUI library in use)

### 1.5 Overview and Honesty Note
Under normal SDLC practice, an SRS is written *before* implementation begins. In this project's actual history, a working prototype was built first, and this SRS is being written retroactively to formally capture what the system does and should do going forward. This is explicitly noted here rather than hidden, because a professional SRS must never misrepresent its own origin — that context matters for anyone maintaining this project later.

---

## 2. Overall Description

### 2.1 Product Perspective
AeroInference is a standalone desktop application (Windows, currently) composed of three internally decoupled layers: a core math/execution engine, a model-compiler/parser layer, and a GUI dashboard. It has no external network dependencies and does not currently integrate with any cloud service, database, or third-party API beyond the GLFW/OpenGL windowing libraries.

### 2.2 Product Functions (Summary)
1. Load a neural network model definition from a JSON file.
2. Validate the model's structural integrity (layer shapes, dimension compatibility).
3. Execute a forward-pass inference given input data.
4. Display real-time latency and status telemetry to the user.
5. (Planned) Quantize model weights from FP32 to INT8.
6. (Planned) Cross-compile and run on an embedded microcontroller target.

### 2.3 User Characteristics
Primary user: a developer/engineer integrating or testing a TinyML model, comfortable with technical dashboards and JSON model files. Not currently designed for non-technical end users.

### 2.4 Constraints
- Must avoid heap allocation during the inference forward pass (performance/determinism requirement for embedded targets).
- Must not use C++ exceptions across `noexcept`-marked boundaries (embedded-safety requirement — see FR-9).
- Current build toolchain is MinGW-w64 g++ on Windows; no cross-platform build has been verified.

### 2.5 Assumptions and Dependencies
- Assumes GLFW and OpenGL 3.3 are available on the host system.
- Assumes model JSON files are well-formed enough to be parsed by the custom lightweight parser (not a full RFC 8259 JSON parser — see FR-3 notes).

---

## 3. Specific Requirements

### 3.1 Functional Requirements

| ID | Requirement | Status |
|---|---|---|
| FR-1 | The system shall load a model definition from a specified JSON file path. | Implemented |
| FR-2 | The system shall construct in-memory layer objects (Conv2D, FullyConnected, Activation_ReLU) from the parsed file, including their weight/bias data. | Implemented |
| FR-3 | The system shall return a descriptive error (not crash) when the model file is missing, malformed, or has mismatched dimensions. | Implemented |
| FR-4 | The system shall validate that consecutive layer shapes are compatible before allowing inference (`compile()`). | Implemented |
| FR-5 | The system shall execute a forward pass through all loaded layers given an input tensor, producing an output tensor. | Implemented |
| FR-6 | The system shall measure and display the wall-clock time taken by a forward pass, in microseconds. | Implemented |
| FR-7 | The system shall provide a GUI control to trigger model loading and display success/failure status to the user. | Implemented |
| FR-8 | The system shall provide a GUI control to trigger a forward pass and display the resulting latency. | Implemented |
| FR-9 | Core execution-path functions shall not throw uncaught C++ exceptions (embedded-safety requirement). | Partially implemented — JSON parsing hardened; not yet audited across all modules |
| FR-10 | The system shall support post-training quantization of model weights from FP32 to INT8. | Interface defined only; logic not yet implemented/verified |
| FR-11 | The system shall support execution on an embedded microcontroller target (not just desktop). | Not started |
| FR-12 | The system shall provide an automated test suite validating math kernel correctness against known reference values. | Implemented — 12/12 tests passing (math kernels, Tensor, Network integration) |

### 3.2 Non-Functional Requirements

| ID | Category | Requirement | Status |
|---|---|---|---|
| NFR-1 | Performance | Forward pass shall execute without heap allocation once the network is compiled. | Implemented (ping-pong buffer design) |
| NFR-2 | Reliability | The application shall not crash (unhandled exception/termination) on malformed input files. | Partially met — one known crash class fixed; full audit pending |
| NFR-3 | Portability | Core engine (`core/`, `compiler/`) shall have no dependency on the GUI layer, to allow future embedded builds without GUI code. | Implemented (verified by folder/module separation) |
| NFR-4 | Maintainability | Codebase shall use consistent, documented interfaces (`ILayer`, `ModelParser`) for extensibility. | Implemented |
| NFR-5 | Testability | Critical math functions shall be independently unit-testable. | Implemented — 12/12 tests passing across math kernels, Tensor, and Network integration |
| NFR-6 | Security | Model files shall not be able to trigger out-of-bounds memory access via malformed dimension data. | Needs verification — not yet fuzz-tested |

### 3.3 External Interface Requirements
- **File I/O:** Reads `.json` model files from a local filesystem path.
- **GUI:** Desktop window (GLFW + OpenGL 3.3 + Dear ImGui), keyboard/mouse input only.
- **Build system:** g++ command-line invocation (see `tasks.json`); no CMake build has been verified despite `CMakeLists.txt` existing in the repository (needs confirmation/testing as a follow-up item).

---

## 4. Open Items for Next Revision
1. Confirm whether `CMakeLists.txt` is actually functional or aspirational (same class of issue as the README's SIMD claims — needs verification, not assumption).
2. Define acceptance criteria for FR-11 (embedded target) once a specific target chip is chosen.
3. Define what "correct" means numerically for FR-12 (e.g., tolerance vs. a NumPy/PyTorch reference implementation).