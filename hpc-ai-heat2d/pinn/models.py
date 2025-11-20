import torch
import torch.nn as nn

class MLP_PINN(nn.Module):
    def __init__(self, in_dim=3, out_dim=1, hidden_dim=64, n_hidden=6):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.Tanh())
        for _ in range(n_hidden - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch_size, 3) containing (x, y, t)
        return self.net(x)
