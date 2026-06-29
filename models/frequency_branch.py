import torch
import torch.nn as nn


class FrequencyBranch(nn.Module):
    """频域分支: FFT幅值谱 → 轻量CNN (detach防反向传播开销)"""
    def __init__(self, out_dim=256):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Dilated conv: 扩大感受野捕捉全局频域模式
            nn.Conv2d(256, 256, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, out_dim)

    def forward(self, x):
        with torch.no_grad():
            # FFT → 幅值谱 (不计算梯度)
            fft = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
            fft_shift = torch.fft.fftshift(fft)
            x_freq = torch.abs(fft_shift)
            x_freq = x_freq / (x_freq.max() + 1e-8)

        x = self.backbone(x_freq)
        x = x.flatten(1)
        x = self.fc(x)
        return x