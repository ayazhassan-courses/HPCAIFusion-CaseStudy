import torch
import numpy as np
from models import MLP_PINN

def evaluate_on_grid(model_path, alpha, Nx, Ny, T, device="cuda"):
    # TODO:
    # 1. create (x, y, t=T) grid
    # 2. load trained model
    # 3. run inference and save u_pred(x,y,T)
    raise NotImplementedError

if __name__ == "__main__":
    # TODO: hook to config or CLI
    pass
