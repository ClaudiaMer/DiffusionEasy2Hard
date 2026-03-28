import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy.random as rnd
from joblib import Parallel, delayed
from scipy.stats import t

VAR = 1.0
VAR1 = 1.0

# ----------------------
# Sampling functions
# ----------------------
def three_mixture(_, a=0.01):
    """Samples from a three-component mixture distribution with mean zero, variance one and non-zero excess kurtosis ."""
    num = len(_)
    y = np.sqrt(1 / (2 * a))
    c = 1 - 2 * a
    x = np.random.uniform(low=-1, high=1.0, size=num)
    mask = (np.abs(x) > c) * 1.0
    return torch.tensor(mask * y * np.sign(x))


def students_t(_, df=4.01):
    """Samples from a Student's t distribution which has mean zero and variance one."""
    num = len(_)
    scale = np.sqrt((df - 2) / df)
    samples = scale * t.rvs(df=df, size=num)
    return torch.tensor(samples)


def make_x(a, z, t=0.3):
    """Compute x = exp(-t) * a + sqrt(1 - exp(-2t)) * z"""
    deltat = 1 - torch.exp(-2 * torch.tensor(t))
    x = torch.exp(-torch.tensor(t)) * a + torch.sqrt(deltat) * z
    return x

def make_noised(d, npoints, npoints_z_over_a=1, t=0.3, device=None,
                sample_first_dim=np.sign):
    """Generate noised samples a, z using PyTorch."""
    a = torch.randn(d, npoints, device=device)
    a[0, :] = sample_first_dim(a[0, :])*np.sqrt(VAR1)
    a[1, :] *= torch.sqrt(torch.tensor(VAR, device=device))
    a[2, :] += 1.0
    a = a.repeat(1, npoints_z_over_a)
    assert a.shape == (d, npoints * npoints_z_over_a)

    z = torch.randn(d, npoints * npoints_z_over_a, device=device)
    x = make_x(a, z, t=t)
    return x.T, z.T




def make_noised_mean(d, npoints, npoints_z_over_a=1, t=0.3, device=None):
    """Generate noised samples with adjusted mean and variance."""
    a = torch.randn(d, npoints, device=device)
    average_variance = (VAR + VAR1 + d - 2) / d
    a *= torch.sqrt(torch.tensor(average_variance, device=device))
    a[2, :] += 1.0
    a = a.repeat(1, npoints_z_over_a)
    assert a.shape == (d, npoints * npoints_z_over_a)

    z = torch.randn(d, npoints * npoints_z_over_a, device=device)
    x = make_x(a, z, t=t)
    return x.T, z.T


def make_noised_mean_cov(d, npoints, npoints_z_over_a=1, t=0.3, device=None):
    """Generate noised samples with mean and covariance adjustments."""
    a = torch.randn(d, npoints, device=device)
    a[0,:] *= np.sqrt(VAR1)
    a[1, :] *= torch.sqrt(torch.tensor(VAR, device=device))
    a[2, :] += 1.0
    a = a.repeat(1, npoints_z_over_a)
    assert a.shape == (d, npoints * npoints_z_over_a)

    z = torch.randn(d, npoints * npoints_z_over_a, device=device)
    x = make_x(a, z, t=t)
    return x.T, z.T