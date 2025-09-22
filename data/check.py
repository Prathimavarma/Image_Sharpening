# check_dataset.py
import os
import cv2

blurry_dir = "blurry"
sharp_dir = "sharp"

def check_images(folder):
    print(f"\nChecking: {folder}")
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        img = cv2.imread(fpath)
        if img is None:
            print(f"❌ Cannot read: {fpath}")
        else:
            print(f"✅ Loaded: {fpath}")

check_images(blurry_dir)
check_images(sharp_dir)
