import torch
import torch.nn as nn
import torch.optim as optim

from models import MLP_PINN
from datasets import sample_interior_points, sample_initial_points, sample_boundary_points

def heat_pde_residual(model, xyt, alpha):
    """
    Compute PDE residual for the heat equation:
        u_t - alpha (u_xx + u_yy) = 0
    using autograd.
    """
    xyt.requires_grad_(True)
    u = model(xyt)              # (N, 1)
    # TODO: use torch.autograd.grad to compute derivatives
    #   u_t, u_xx, u_yy
    # residual = u_t - alpha * (u_xx + u_yy)
    raise NotImplementedError

def train(config):
    device = config.get("device", "cuda")
    alpha = config["alpha"]

    model = MLP_PINN(
        in_dim=3,
        out_dim=1,
        hidden_dim=config["hidden_dim"],
        n_hidden=config["n_hidden"]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    # TODO: optional: learning rate scheduler

    for epoch in range(config["max_epochs"]):
        model.train()
        optimizer.zero_grad()

        # Sample collocation points
        xyt_pde = sample_interior_points(config["N_pde"], device=device)
        xyt_ic, u_ic = sample_initial_points(config["N_ic"], device=device)
        xyt_bc, u_bc = sample_boundary_points(config["N_bc"], device=device)

        # PDE loss
        resid = heat_pde_residual(model, xyt_pde, alpha)
        loss_pde = (resid**2).mean()

        # IC loss
        u_pred_ic = model(xyt_ic)
        loss_ic = ((u_pred_ic - u_ic)**2).mean()

        # BC loss
        u_pred_bc = model(xyt_bc)
        loss_bc = ((u_pred_bc - u_bc)**2).mean()

        loss = (config["w_pde"] * loss_pde
                + config["w_ic"] * loss_ic
                + config["w_bc"] * loss_bc)

        loss.backward()
        optimizer.step()

        if epoch % config.get("log_interval", 100) == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4e} "
                  f"(PDE: {loss_pde.item():.4e}, IC: {loss_ic.item():.4e}, BC: {loss_bc.item():.4e})")

        # TODO: add checkpoint saving, early stopping, TensorBoard logging

    # TODO: save trained model to file
    torch.save(model.state_dict(), config["model_out_path"])

if __name__ == "__main__":
    # TODO: load config from YAML or argparse
    example_config = {
        "alpha": 0.1,
        "hidden_dim": 64,
        "n_hidden": 6,
        "lr": 1e-3,
        "max_epochs": 10000,
        "N_pde": 10000,
        "N_ic": 1000,
        "N_bc": 1000,
        "w_pde": 1.0,
        "w_ic": 1.0,
        "w_bc": 1.0,
        "log_interval": 500,
        "model_out_path": "../data/pinn_outputs/pinn_heat2d.pt",
        "device": "cuda"
    }
    train(example_config)
