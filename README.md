# Image_Sharpening
Image Sharpening using Knowledge Distillation – Enhanced image quality using deep learning and knowledge distillation while reducing model complexity.
The workflow demonstrates how Knowledge Distillation (KD) can transfer performance from large deep networks to smaller, faster models — making them efficient for deployment.

 Features:

✅ Dataset preparation with sharp & blurry image pairs

✅ Teacher model (ResNet34) for supervision

✅ Student model (lightweight CNN) for fast inference

✅ Knowledge Distillation + Perceptual Loss training

✅ Evaluation using PSNR and SSIM

✅ Visualization of blurry vs sharpened vs ground truth images

 Project Structure :
image_sharpening_kd/
│
├── data/
│   ├── blurry/       # Input blurry images
│   └── sharp/        # Ground truth sharp images
│
├── models/
│   ├── teacher.py    # Teacher model (ResNet-based)
│   └── student.py    # Student model (lightweight CNN)
│
├── losses/
│   ├── perceptual_loss.py
│   └── distillation_loss.py
│
├── train.py          # Training loop
├── evaluate.py       # Evaluation & visualization
├── utils.py          # Helper functions (dataloader, metrics, etc.)
└── requirements.txt  # Dependencies


Install dependencies:

pip install -r requirements.txt

🗂️ Dataset

Place your blurry images in data/blurry/

Place corresponding sharp images in data/sharp/

👉 Example:

data/
 ├── blurry/
 │    ├── img1.png
 │    ├── img2.png
 │    └── ...
 └── sharp/
      ├── img1.png
      ├── img2.png
      └── ...

🏋️ Training

Run training with:

python train.py

📊 Evaluation

Evaluate model performance (PSNR & SSIM) and visualize outputs:

python evaluate.py


Example output:

PSNR: 13.82, SSIM: 0.459

🖼️ Sample Results
Blurry Input	Sharpened Output	Ground Truth

	
	
📌 Future Work

Add support for larger datasets

Improve student model with attention mechanisms

Deploy as a lightweight web app

👩‍💻 Author

Prathima Varma
📌 Project for exploring deep learning, computer vision, and knowledge distillation.
