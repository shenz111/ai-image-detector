import torch


def fft_magnitude(x):
    """FFT幅值谱 (B, C, H, W) → (B, C, H, W)"""
    fft = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
    fft_shift = torch.fft.fftshift(fft)
    magnitude = torch.abs(fft_shift)
    return magnitude / (magnitude.max() + 1e-8)