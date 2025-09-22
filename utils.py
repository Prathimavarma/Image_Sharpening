import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Custom Dataset for Blurry → Sharp pairs
class ImagePairDataset(Dataset):
    def __init__(self, blurry_dir, sharp_dir, transform=None):
        self.blurry_dir = blurry_dir
        self.sharp_dir = sharp_dir
        self.transform = transform

        # Match images by filename
        self.images = os.listdir(blurry_dir)
        self.images = [img for img in self.images if os.path.exists(os.path.join(sharp_dir, img))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        blurry_path = os.path.join(self.blurry_dir, img_name)
        sharp_path = os.path.join(self.sharp_dir, img_name)

        blurry_img = cv2.imread(blurry_path)
        sharp_img = cv2.imread(sharp_path)

        # Convert to RGB
        blurry_img = cv2.cvtColor(blurry_img, cv2.COLOR_BGR2RGB)
        sharp_img = cv2.cvtColor(sharp_img, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image for transforms
        blurry_img = transforms.ToPILImage()(blurry_img)
        sharp_img = transforms.ToPILImage()(sharp_img)

        if self.transform:
            blurry_img = self.transform(blurry_img)
            sharp_img = self.transform(sharp_img)

        return blurry_img, sharp_img

# Image Transform (Resize, Normalize)
def get_transform(image_size=256):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

# PSNR & SSIM Metrics
def calculate_psnr(img1, img2):
    """Compute PSNR between two images (numpy arrays)."""
    return psnr(img1, img2, data_range=img2.max() - img2.min())


def calculate_ssim(img1, img2):
    """Compute SSIM between two images (numpy arrays)."""
    return ssim(img1, img2, channel_axis=2, data_range=img2.max() - img2.min())

# Get Dataloader
def get_dataloader(blurry_dir, sharp_dir, batch_size=4, image_size=256, shuffle=True):
    transform = get_transform(image_size)
    dataset = ImagePairDataset(blurry_dir, sharp_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return dataloader
