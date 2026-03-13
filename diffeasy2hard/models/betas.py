import torch 
import numpy as np


def linear(mi=1e-4, ma=2e-2, T=1000): 
    """Values of beta_t, from Ho et al. 

    Args:
        mi (float, optional): Minimum beta value. Defaults to 1e-4.
        ma (float, optional): Maximum beta value. Defaults to 2e-2.
        T (int, optional): Number of time steps in diffusion model. Defaults to 1000.

    Returns:
        torch.Tensor: beta values for t=1 to T
    """
    if T ==1000:
        ts = torch.arange(T)
        betas = (ma-mi)/(T-1)*ts + mi
        torch.linspace(ma, mi, T, dtype=torch.float32)
    elif T == 10:
        ts = torch.arange(T)/T
        betas = ts
    else: 
        betas = 0.01*torch.ones(T)
    return betas

def linear_scaled(mi=1e-4, ma=2e-2, T=1000): 
    """Values of beta_t, from Ho et al, but rescale mi and ma by 1000/T to keep the same noise scale for different T.

    Args:
        mi (float, optional): Minimum beta value. Defaults to 1e-4.
        ma (float, optional): Maximum beta value. Defaults to 2e-2.
        T (int, optional): Number of time steps in diffusion model. Defaults to 1000.

    Returns:
        torch.Tensor: beta values for t=1 to T
    """
    ma = min(0.9,ma*1000/T)
    mi = max(mi, mi*1000/T)
    intermediate = 1.0/T
    betas = torch.zeros(T)
    betas[:T//2] = torch.linspace(mi, intermediate, T//2)
    betas[T//2:] = torch.linspace(intermediate, ma, T//2)
    return betas