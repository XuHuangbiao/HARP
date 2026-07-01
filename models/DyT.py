import torch
from torch import nn


class DyT(nn.Module):
    def __init__(self, num_features, init_alpha=0.6):
        super(DyT, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))


    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return self.gamma * x + self.beta
