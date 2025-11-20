#pragma once

#include <vector>

// Simple struct to hold simulation parameters
struct Heat2DParams {
    int Nx;        // grid points in x
    int Ny;        // grid points in y
    int nSteps;    // number of time steps
    float alpha;   // thermal diffusivity
    float dx;      // spatial step (assume dx = dy)
    float dt;      // time step
};

// Host-side API to run the simulation
void run_heat2d_gpu(const Heat2DParams& params,
                    std::vector<float>& u_host,
                    bool save_intermediate = false,
                    const std::string& out_dir = "");
