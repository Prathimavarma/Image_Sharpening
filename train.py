# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models.teacher import TeacherModel
from models.student import StudentModel
from losses.perceptual_loss import PerceptualLoss
from losses.distillation_loss import DistillationLoss
from utils import get_dataloader

device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
epochs = 5
batch_size = 2
lr = 1e-4
image_size = 256

# Data
train_loader = get_dataloader("data/blurry", "data/sharp", batch_size=batch_size, image_size=image_size)

# Models
teacher = TeacherModel().to(device).eval()
student = StudentModel().to(device)

# Losses
l1_loss = nn.L1Loss()
perc_loss = PerceptualLoss().to(device)
distill_loss = DistillationLoss().to(device)

# Optimizer
optimizer = optim.Adam(student.parameters(), lr=lr)

# Training Loop
for epoch in range(epochs):
    student.train()
    total_loss = 0
    for blurry, sharp in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        blurry, sharp = blurry.to(device), sharp.to(device)

        with torch.no_grad():
            teacher_out, teacher_feats = teacher(blurry)

        student_out, student_feats = student(blurry)

        loss_r = l1_loss(student_out, sharp)
        loss_p = perc_loss(student_out, sharp)
        loss_d = distill_loss([student_feats[-1]], [teacher_feats])

        loss = loss_r + 0.1 * loss_p + 0.05 * loss_d

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

torch.save(student.state_dict(), "student_model.pth")
print("Training Complete! Model saved as student_model.pth")
