cmake_minimum_required(VERSION 3.18)
project(heat2d_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

add_executable(heat2d
    src/main.cu
    src/heat2d_kernel.cu
)
target_include_directories(heat2d PRIVATE include)

# TODO: set GPU arch flags, e.g.:
# set_target_properties(heat2d PROPERTIES
#     CUDA_ARCHITECTURES "80"
# )
