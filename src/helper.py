import cv2
import os
from pathlib import Path
# load images and return a tuple of images and skipped images

def load_image(fold_path : str) -> tuple[list, list]:
    imgs = []
    skipped_imgs = []

    for filename in os.listdir(fold_path):
        image_path = os.path.join(fold_path, filename)

        try:
           img = cv2.imread(image_path)
           if img is None:
                raise ValueError("OpenCV could not decode this file")

           imgs.append(img)
        except Exception as e:
            print(f"WARNING: Could not read {image_path}: {e}")
            skipped_imgs.append(image_path)

    print(f"Successfully loaded: {len(imgs)} images")
    print(f"Skipped: {len(skipped_imgs)} files")
    return imgs, skipped_imgs