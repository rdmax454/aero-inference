#include "DashboardApp.hpp"

/**
 * @brief Master system framework entry point.
 */
int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    DashboardApp app;
    
    // Set up standard 1280x720 workspace resolution
    if (!app.initialize(1280, 720, "AeroInference Developer Workstation Console")) {
        return -1;
    }

    app.run_loop();
    return 0;
}