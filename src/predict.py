import os

import cv2
import numpy as np

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def predict_image(path, model, classes, img_size=(224, 224), min_conf=0.60):
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError(f"path must be a str or PathLike, got {type(path).__name__}")
        path = os.fspath(path)
    if not path.strip():
        raise ValueError("path is empty")
    if not os.path.exists(path):
        raise FileNotFoundError(f"file not found: {path}")
    if os.path.isdir(path):
        raise IsADirectoryError(f"this is a folder, not an image file: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext not in ALLOWED_EXT:
        raise ValueError(f"unsupported file type '{ext}' (allowed: {', '.join(sorted(ALLOWED_EXT))})")
    if not os.access(path, os.R_OK):
        raise PermissionError(f"no permission to read: {path}")
    if os.path.getsize(path) == 0:
        raise ValueError(f"file is empty: {path}")

    img = cv2().imread(path)
    if img is None:
        raise ValueError(f"OpenCV could not decode this file (corrupt or not a real image): {path}")

    img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
    x = img.astype("float32")[None] / 255.0

    probs = model.predict(x, verbose=0)[0]
    k = int(np.argmax(probs))
    return {
        "label": classes[k],
        "confidence": float(probs[k]),
        "probs": {c: float(p) for c, p in zip(classes, probs)},
        "uncertain": float(probs[k]) < min_conf,
        "image": img,  # resized BGR image, for display
    }

def safe_predict(path, model, classes, **kwargs):
    try:
        return predict_image(path, model, classes, **kwargs)
    except (TypeError, ValueError, FileNotFoundError, IsADirectoryError, PermissionError) as e:
        print(f"Cannot predict: {type(e).__name__}: {e}")
        return None

def show_prediction(result, title=None):
    import matplotlib as plt
    if result is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].imshow(result["image"][..., ::-1])  # BGR -> RGB
    ax[0].axis("off")
    ax[0].set_title(title or f"{result['label']} ({result['confidence']:.1%})")
    ax[1].barh(list(result["probs"].keys()), list(result["probs"].values()))
    ax[1].set_xlim(0, 1)
    ax[1].set_title("class probabilities")
    plt.tight_layout()
    plt.show()
    if result["uncertain"]:
        print("Low confidence: treat this prediction as unreliable.")