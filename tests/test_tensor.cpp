#include "Tensor.hpp"
#include <iostream>
#include <cmath>

bool nearly_equal(float a, float b, float epsilon = 0.0001f) {
    return std::fabs(a - b) < epsilon;
}

// =============================================================================
// Test 1: Basic construction and element access
// =============================================================================
bool test_tensor_basic_access() {
    Tensor t({2, 3}); // 2 rows, 3 columns = 6 elements total

    if (t.size() != 6) return false;

    // A freshly constructed Tensor must be zero-initialized
    for (size_t i = 0; i < t.size(); ++i) {
        if (!nearly_equal(t[i], 0.0f)) return false;
    }

    // Write and read back a value directly by flat index
    t[0] = 42.0f;
    if (!nearly_equal(t[0], 42.0f)) return false;

    return true;
}

// =============================================================================
// Test 2: Stride-based coordinate lookup
// For a {2, 3} shaped tensor (row-major), the correct flat offset formula is:
//   offset = row * 3 + col
// So coordinate (1, 2) -> offset = 1*3 + 2 = 5
// =============================================================================
bool test_tensor_coordinate_offset() {
    Tensor t({2, 3});

    size_t offset = t.calculate_flat_offset({1, 2});
    if (offset != 5) return false;

    size_t offset_origin = t.calculate_flat_offset({0, 0});
    if (offset_origin != 0) return false;

    return true;
}

// =============================================================================
// Test 3: Move semantics — Tensor should transfer ownership cleanly
// =============================================================================
bool test_tensor_move_semantics() {
    Tensor original({2, 2});
    original[0] = 99.0f;

    Tensor moved = std::move(original);

    if (!nearly_equal(moved[0], 99.0f)) return false;
    if (moved.size() != 4) return false;

    // After a move, the original should no longer claim ownership of the data
    if (original.size() != 0) return false;

    return true;
}

int main() {
    int passed = 0;
    int total = 0;

    auto run = [&](const char* name, bool (*test_fn)()) {
        total++;
        if (test_fn()) {
            std::cout << "[PASS] " << name << "\n";
            passed++;
        } else {
            std::cout << "[FAIL] " << name << "\n";
        }
    };

    run("test_tensor_basic_access", test_tensor_basic_access);
    run("test_tensor_coordinate_offset", test_tensor_coordinate_offset);
    run("test_tensor_move_semantics", test_tensor_move_semantics);

    std::cout << "\n" << passed << "/" << total << " tests passed.\n";
    return (passed == total) ? 0 : 1;
}