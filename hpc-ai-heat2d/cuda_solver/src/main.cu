#include "heat2d.hpp"
#include <iostream>
#include <vector>
#include <string>

// TODO: you may add a CLI parsing library or keep it simple

int main(int argc, char** argv)
{
    // TODO: parse command-line args:
    //   Nx, Ny, nSteps, alpha, dx, dt, output directory, etc.
    Heat2DParams params;
    params.Nx = 512;
    params.Ny = 512;
    params.nSteps = 1000;
    params.alpha = 0.1f;
    params.dx = 1.0f / (params.Nx - 1);
    params.dt = 1e-4f;

    std::cout << "Running 2D heat equation on GPU with:\n"
              << "Nx = " << params.Nx << ", Ny = " << params.Ny
              << ", nSteps = " << params.nSteps << std::endl;

    // Allocate and set initial condition on host
    std::vector<float> u_host(params.Nx * params.Ny, 0.0f);

    // TODO: initialize u_host with chosen IC (e.g., Gaussian hot spot)

    // Run simulation
    run_heat2d_gpu(params, u_host, /*save_intermediate=*/true,
                   /*out_dir=*/"../data/cuda_reference");

    // TODO: write final u_host to disk (e.g., .npy, .csv, or binary)

    std::cout << "Simulation finished.\n";
    return 0;
}
