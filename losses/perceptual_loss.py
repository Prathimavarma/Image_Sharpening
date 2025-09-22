# losses/perceptual_loss.py
import torch
import torch.nn as nn
import torchvision.models as models

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        self.layers = nn.Sequential(*list(vgg)[:16])
        for p in self.layers.parameters():
            p.requires_grad = False
        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        fx = self.layers(x)
        fy = self.layers(y)
        return self.criterion(fx, fy)
