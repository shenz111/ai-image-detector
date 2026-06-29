import torch.nn as nn
from models.spatial_branch import SpatialBranch
from models.frequency_branch import FrequencyBranch
from models.fusion import FusionModule
from models.classifier import Classifier


class AIDetector(nn.Module):
    """
    双分支架构: ResNet34(空间) + FFT轻量CNN(频域) + 门控融合
    """
    def __init__(self):
        super().__init__()

        self.spatial = SpatialBranch(out_dim=256)
        self.frequency = FrequencyBranch(out_dim=256)
        self.fusion = FusionModule(dim=256)
        self.classifier = Classifier(dim=256)

    def forward(self, x):
        s_feat = self.spatial(x)
        f_feat = self.frequency(x)

        fused = self.fusion(s_feat, f_feat)
        out = self.classifier(fused)

        return out
