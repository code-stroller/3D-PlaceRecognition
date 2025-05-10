# models/ours/pooling.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class MAC(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        # Return (batch_size, n_features)
        return torch.max(x, dim=-1, keepdim=False)[0]


class SPoC(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        # Return (batch_size, n_features)
        return torch.mean(x, dim=-1, keepdim=False)


class GeM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.p   = nn.Parameter(torch.ones(1) * cfg.p)
        self.eps = cfg.eps

    def forward(self, x):
        # 1) clamp negative values to eps, then raise to the power p
        x = x.clamp(min=self.eps).pow(self.p)

        # 2) global average pooling over N points
        x = torch.mean(x, dim=-1, keepdim=False)

        # 3) take the p-th root
        return x.pow(1.0 / self.p)
