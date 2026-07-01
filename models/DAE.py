import torch
from torch import nn
from models.DyT import DyT
from models.AttnW import FeatDimAttnW


class DAE(nn.Module):
    def __init__(self, args, in_dae, dropout):
        super(DAE, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dae, in_dae // 2),
            nn.GELU(),
            nn.Linear(in_dae // 2, in_dae // 4),
            nn.GELU(),
            nn.Linear(in_dae // 4, in_dae // 8),
            DyT(in_dae // 8),
        )
        self.fc_mean = nn.Linear(in_dae // 8, 1)
        self.fc_logvar = nn.Linear(in_dae // 8, 1)

    def encode(self, x):
        x = self.mlp(x)
        return self.fc_mean(x), self.fc_logvar(x)

    def reparametrization(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn_like(mu)
        scores = mu + std * esp
        return scores, std

    def forward(self, x):
        mu, logvar = self.encode(x)
        scores, std = self.reparametrization(mu, logvar)
        return scores, mu, std