import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiMLP(nn.Module):
    def __init__(self, input_dim, output_dim, num_iter=3):
        self.layers = model = [Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1D(output_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        ) for _ in range(num_iter)]
    
    def forward(self, x):
        x = 0
        [(x += layer(x)) for layer in range(self.layers)]
