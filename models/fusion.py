import torch
import torch.nn as nn


class FusionModule(nn.Module):
    """特征级门控融合: 每个维度独立权重"""
    def __init__(self, dim=256):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim * 2),
            nn.Sigmoid(),
        )

    def forward(self, s, f):
        x = torch.cat([s, f], dim=1)
        w = self.gate(x)                     # (B, 512) per-dimension weights
        w_s, w_f = w.chunk(2, dim=1)         # (B, 256), (B, 256)
        return s * w_s + f * w_f