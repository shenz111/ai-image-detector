import torch
import torch.nn as nn
import torchvision.models as models


class SpatialBranch(nn.Module):
    """空间分支: ResNet34 多尺度特征 (轻量高效)"""
    def __init__(self, out_dim=256):
        super().__init__()

        backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1  # 64
        self.layer2 = backbone.layer2  # 128
        self.layer3 = backbone.layer3  # 256
        self.layer4 = backbone.layer4  # 512

        # 多尺度投影 (加入 layer1 浅层特征)
        self.proj1 = nn.Linear(64, out_dim)
        self.proj2 = nn.Linear(128, out_dim)
        self.proj3 = nn.Linear(256, out_dim)
        self.proj4 = nn.Linear(512, out_dim)

        self.weights = nn.Parameter(torch.ones(4) / 4)

        self.fusion = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.stem(x)
        f1 = self.layer1(x)  # 64
        f2 = self.layer2(f1)  # 128
        f3 = self.layer3(f2)  # 256
        f4 = self.layer4(f3)  # 512

        p1 = self.proj1(f1.mean([2, 3]))
        p2 = self.proj2(f2.mean([2, 3]))
        p3 = self.proj3(f3.mean([2, 3]))
        p4 = self.proj4(f4.mean([2, 3]))

        # 四尺度可学习加权融合
        w = torch.softmax(self.weights, dim=0)
        out = p1 * w[0] + p2 * w[1] + p3 * w[2] + p4 * w[3]

        return self.fusion(out)