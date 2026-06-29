import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, dim=256):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3), # 随机丢弃防过拟合
            nn.Linear(128, 2)   # 128维 → 2维（输出两类分数）
        )

    def forward(self, x):
        return self.mlp(x)