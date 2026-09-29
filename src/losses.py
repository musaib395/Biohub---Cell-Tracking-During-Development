import torch
import torch.nn as nn

class WeightedHeatmapLoss(nn.Module):
    def __init__(self, alpha=20.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, prediction, target):
        weight = 1.0 + self.alpha * target
        loss = weight * (prediction - target) ** 2
        return loss.mean()