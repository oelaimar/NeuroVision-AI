
import os
import sys

# Allow Python to find modules inside src/
sys.path.append('../')
from src.helper import load_image

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


def test_load_images():
    for class_name in CLASSES:
        folder_path = os.path.join(RAW_DIR, class_name)

        assert os.path.isdir(folder_path), (
            f"Folder not found: {folder_path}"
        )

        images, skipped_images = load_image(folder_path)

        print(f"\nClass: {class_name}")
        print(f"Loaded: {len(images)}")
        print(f"Skipped: {len(skipped_images)}")

        for img in images:
            assert img is not None
            assert len(img.shape) == 3
            assert img.shape[2] == 3

    print("\nAll image-loading checks passed!")


if __name__ == "__main__":
    test_load_images()
