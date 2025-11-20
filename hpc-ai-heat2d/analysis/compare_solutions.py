import numpy as np
import matplotlib.pyplot as plt
from utils_io import load_cuda_field
# TODO: import PINN evaluation

def compute_l2_error(ref, pred):
    diff = ref - pred
    return np.sqrt(np.sum(diff**2) / np.sum(ref**2))

def main():
    # TODO: paths to CUDA and PINN fields at final time
    # cuda_field = load_cuda_field(...)
    # pinn_field = load_pinn_field(...)
    # err = compute_l2_error(cuda_field, pinn_field)
    # print("Relative L2 error:", err)

    # TODO: plot heatmaps and error map

    pass

if __name__ == "__main__":
    main()
