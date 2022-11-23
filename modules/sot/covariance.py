import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
from torch_scatter import scatter_softmax, scatter_mean
class Covariance(nn.Module):
    def __init__(self, remove_mean=True):
        super(Covariance, self).__init__()
        self.remove_mean = remove_mean

    def _remove_mean(self, x, batch):
        # x [H, M, D] 
        _mean = scatter_mean(x, batch, dim=1)
        _mean = _mean[:,batch,:]
        x = x - _mean
        return x

    def cov(self, x, batch):
        H, M, D = x.size()
        x = torch.einsum("hik,hkj->hkij", x.permute(0, 2, 1), x)
        y = scatter_mean(x, batch, dim=1)
        return y
    
    
    def cross_cov(self, x1, x2, batch):
        H1, M1, D1 = x1.size()
        H2, M2, D2 = x2.size()
        y = torch.einsum("hik,hkj->hkij", x1.permute(0, 2, 1), x2)
        y = scatter_mean(y, batch, dim=1)
        return y

    def forward(self, batch, x, y=None):
        if self.remove_mean:
            x = self._remove_mean(x, batch)
            if y is not None:
                y = self._remove_mean(y, batch)
        if y is not None:
            x = self.cross_cov(x, y, batch)
        else:
            x = self.cov(x, batch)
        return x
