#include "heat2d.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

// TODO: adjust as needed
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16

// Device kernel: one time-step update using 5-point stencil
__global__
void heat2d_step_kernel(const float* __restrict__ u_old,
                        float* __restrict__ u_new,
                        int Nx, int Ny,
                        float alpha_dt_dx2)
{
    // TODO 1: compute (i, j) from blockIdx, threadIdx
    // TODO 2: handle interior points only; skip boundary
    // TODO 3: load neighbors and apply 5-point stencil
    //   u_new = u_old + alpha_dt_dx2 * (sum of neighbors - 4*u_old)
    // NOTE: ensure memory access is coalesced (row-major)
}

// Optional: shared-memory tiled version (advanced)
__global__
void heat2d_step_kernel_tiled(const float* __restrict__ u_old,
                              float* __restrict__ u_new,
                              int Nx, int Ny,
                              float alpha_dt_dx2)
{
    // TODO (advanced):
    //   - declare shared memory tile with halo
    //   - load tile from global memory
    //   - __syncthreads()
    //   - compute stencil using shared memory
}


void run_heat2d_gpu(const Heat2DParams& params,
                    std::vector<float>& u_host,
                    bool save_intermediate,
                    const std::string& out_dir)
{
    const int Nx = params.Nx;
    const int Ny = params.Ny;
    const size_t N = static_cast<size_t>(Nx) * Ny;
    const size_t bytes = N * sizeof(float);

    // Sanity check
    if (static_cast<int>(u_host.size()) != Nx * Ny) {
        throw std::runtime_error("u_host size does not match Nx*Ny.");
    }

    // Compute alpha_dt_dx2 = alpha * dt / dx^2
    float alpha_dt_dx2 = params.alpha * params.dt / (params.dx * params.dx);

    // TODO: check CFL stability (for explicit scheme) and print warning

    // Allocate device memory
    float *d_u_old = nullptr, *d_u_new = nullptr;
    cudaMalloc(&d_u_old, bytes);
    cudaMalloc(&d_u_new, bytes);

    // Copy initial condition to device
    cudaMemcpy(d_u_old, u_host.data(), bytes, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 gridDim((Nx + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
                 (Ny + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);

    // TODO: add CUDA events for timing
    for (int step = 0; step < params.nSteps; ++step) {
        // Launch kernel
        heat2d_step_kernel<<<gridDim, blockDim>>>(
            d_u_old, d_u_new, Nx, Ny, alpha_dt_dx2
        );

        // TODO: optional: save intermediate snapshots every k steps
        //   - copy back to host
        //   - write to file in out_dir

        // Swap pointers
        std::swap(d_u_old, d_u_new);
    }

    // Copy final result back
    cudaMemcpy(u_host.data(), d_u_old, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_u_old);
    cudaFree(d_u_new);
}
