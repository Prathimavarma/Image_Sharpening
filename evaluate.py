# evaluate.py
import torch
import matplotlib.pyplot as plt
import numpy as np
from models.student import StudentModel
from utils import get_dataloader, calculate_psnr, calculate_ssim

device = "cuda" if torch.cuda.is_available() else "cpu"

# Helper function to convert tensors to images
def tensor_to_image(tensor):
    """Convert a normalized tensor (-1,1) to a displayable image (0,1)."""
    img = tensor.squeeze().detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
    img = (img * 0.5) + 0.5             # De-normalize [-1,1] -> [0,1]
    img = np.clip(img, 0, 1)
    return img

# Load trained student model
student = StudentModel().to(device)
student.load_state_dict(torch.load("student_model.pth", map_location=device))
student.eval()

# Load test dataset
test_loader = get_dataloader("data/blurry", "data/sharp", batch_size=1, image_size=256, shuffle=False)

# Run Evaluation
for blurry, sharp in test_loader:
    blurry, sharp = blurry.to(device), sharp.to(device)

    with torch.no_grad():
        pred, _ = student(blurry)

    blurry_img = tensor_to_image(blurry)
    sharp_img = tensor_to_image(sharp)
    pred_img = tensor_to_image(pred)

    psnr_score = calculate_psnr(pred_img, sharp_img)
    ssim_score = calculate_ssim(pred_img, sharp_img)

    print(f"PSNR: {psnr_score:.2f}, SSIM: {ssim_score:.3f}")

    # Display side-by-side images
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(blurry_img)
    plt.title("Blurry")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(pred_img)
    plt.title("Sharpened (Student)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(sharp_img)
    plt.title("Ground Truth")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
    break
