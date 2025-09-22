# losses/distillation_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self):
        super(DistillationLoss, self).__init__()
        self.criterion = nn.L1Loss()
        self.adapters = nn.ModuleDict()  # To store 1x1 convs dynamically

    def forward(self, student_feats, teacher_feats):
        loss = 0
        for i, (sf, tf) in enumerate(zip(student_feats, teacher_feats)):
            # Resize teacher feature to match student feature spatial size
            if sf.shape[2:] != tf.shape[2:]:
                tf = F.interpolate(tf, size=sf.shape[2:], mode='bilinear', align_corners=False)

            # Match channel sizes dynamically
            if sf.shape[1] != tf.shape[1]:
                key = f"adapter_{i}"
                if key not in self.adapters:
                    self.adapters[key] = nn.Conv2d(tf.shape[1], sf.shape[1], kernel_size=1).to(tf.device)
                adapter = self.adapters[key]
                tf = adapter(tf)

            loss += self.criterion(sf, tf.detach())
        return loss
