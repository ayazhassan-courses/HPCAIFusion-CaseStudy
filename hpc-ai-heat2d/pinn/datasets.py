import torch

def sample_interior_points(N_pde, device="cuda"):
    # TODO: sample (x, y, t) uniformly inside domain (0,1)x(0,1)x(0,T)
    # return tensor of shape (N_pde, 3)
    raise NotImplementedError

def sample_initial_points(N_ic, device="cuda"):
    # TODO: sample (x, y, t=0) and compute u0(x, y)
    # return xyt_ic, u_ic
    raise NotImplementedError

def sample_boundary_points(N_bc, device="cuda"):
    # TODO: sample (x, y, t) on boundary of spatial domain
    # return xyt_bc, u_bc (often 0)
    raise NotImplementedError
