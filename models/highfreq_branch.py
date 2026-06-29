import torch.nn as nn
from utils.fft_utils import laplacian_filter


class HighFreqBranch(nn.Module):
    """高频边缘分支: Laplacian高通滤波 → 轻量CNN (从头训练)"""
    def __init__(self, out_dim=128):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                  # 112

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                  # 56

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                  # 28

            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        x = laplacian_filter(x)       # (B, 3, H, W) 高频边缘
        x = self.backbone(x)          # (B, 128, 1, 1)
        x = x.flatten(1)              # (B, 128)
        x = self.fc(x)
        return x
