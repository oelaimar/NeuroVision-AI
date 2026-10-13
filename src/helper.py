
from PIL import Image
import os
# load images and return a tuple of images and skipped images

def load_image(folder_path : str) -> tuple[list, list]:
    images = []
    skipped_images = []

    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)

        try:
            with Image.open(image_path) as img:
                images.append(img.convert("RGB").copy())

        except Exception as e:
            print(f"WARNING: Could not read {image_path}: {e}")
            skipped_images.append(image_path)
    print(f"Successfully loaded: {len(images)} images")
    print(f"Skipped: {len(skipped_images)} images")
    return images, skipped_images