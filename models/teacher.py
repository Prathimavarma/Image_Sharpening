# models/teacher.py
import torch
import torch.nn as nn
import torchvision.models as models

class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        # Using pretrained ResNet as a feature extractor (proxy for teacher)
        backbone = models.resnet34(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        self.refine = nn.Conv2d(512, 3, kernel_size=1)

    def forward(self, x):
        features = self.feature_extractor(x)
        out = self.refine(features)
        out = torch.nn.functional.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out, features
