import torch 
from torch import nn 
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np


class TwoLayer(nn.Module):
    def __init__(self, d, m, act=F.tanh, bias=True, t=0.3, wnorm=1.0):
        """
        Two-layer network with tied weights:
        - Second layer uses the transpose of the first layer's weights.

        Parameters
        ----------
        d : int
            Input dimension
        m : int
            Hidden dimension (number of hidden nodes)
        """
        super().__init__()
        self.m = m
        self.g = act
        self.d = d
        self.delta = np.sqrt(1-np.exp(-2*t))
        self.c = np.exp(-t)/self.delta

        # First layer has its own weight and bias
        self.fc1 = nn.Linear(d,m, bias=False)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.1/np.sqrt(d))
        # Second layer only has its bias (weights are tied to fc1)
        self.fc2_bias = torch.zeros(d)  # learnable bias for the second layer
        self.scale = nn.Parameter(torch.ones(1))
        self.scale.requires_grad = True

    def forward(self, x):
        # First layer
        hidden = self.g(self.fc1(self.c*x))
        # Second layer: manually apply tied weights
        # y = hidden @ W1^T + b2
        out = self.scale*x + F.linear(hidden, self.fc1.weight.t(), self.fc2_bias)
        return out
    
    def norm_transpose_weight(self): 
        with torch.no_grad(): 
            lengths = self.fc1.weight.norm(dim=1)
            assert len(lengths) == self.m
            self.fc1.weight /= lengths




class NLayer(nn.Module):
    def __init__(self, d, m, act=F.tanh, bias=True, t=0.3, wnorm=1.0, N=3,skip=1.0):
        """
        Two-layer network with tied weights:
        - Second layer uses the transpose of the first layer's weights.

        Parameters
        ----------
        d : int
            Input dimension
        m : int
            Hidden dimension (number of hidden nodes)
        """
        super().__init__()
        self.m = m
        self.g = act
        self.d = d
        self.delta = np.sqrt(1-np.exp(-2*t))
        self.c = np.exp(-t)/self.delta

        # First layer has its own weight and bias
        self.fc1 = nn.Linear(d,m, bias=False)
        self.layers = []
        if N>2: 
            for n in range(N-2): 
                self.layers += [nn.Linear(m,m, bias=False)]
        self.out = nn.Linear(m, d, bias=False)

        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.1/np.sqrt(d))
        # Second layer only has its bias (weights are tied to fc1)
        self.fc2_bias = torch.zeros(d)  # learnable bias for the second layer
        self.scale = nn.Parameter(torch.ones(1))
        self.scale.requires_grad = False
        self.skip=skip

    def forward(self, x):
        # First layer
        nonlin = self.g(self.fc1(x))
        for layer in self.layers:
            nonlin_ = layer(nonlin)
            nonlin_ = self.g(nonlin_)
            nonlin = self.skip*nonlin + nonlin_
        nonlin = self.out(nonlin)
        # Second layer: manually apply tied weights
        # y = hidden @ W1^T + b2
        out = self.scale*x + nonlin
        return out
    
    def norm_transpose_weight(self): 
        with torch.no_grad(): 
            lengths = self.fc1.weight.norm(dim=1)
            assert len(lengths) == self.m
            self.fc1.weight /= lengths


