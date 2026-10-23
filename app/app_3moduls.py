from pathlib import Path

import cv2
import keras
import numpy as np
import streamlit as st
import tensorflow as tf

# --- compatibility patch: models saved with a newer Keras add `quantization_config` ---
if not getattr(keras.layers.Layer, "_qc_patched", False):
    _orig_layer_init = keras.layers.Layer.__init__

    def _patched_layer_init(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        _orig_layer_init(self, *args, **kwargs)

    keras.layers.Layer.__init__ = _patched_layer_init
    keras.layers.Layer._qc_patched = True
# -------------------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
RUNS_DIR = BASE_DIR / "runs" / "20261022-152441_mobilenetv2_seed42" # need to change this every time

# name shown in the UI -> model file (edit these)
MODEL_PATHS = {
    "CNN": MODEL_DIR / "mri_cnn.keras",
    "best_head": RUNS_DIR / "best_head.keras",
    "fine-tune": RUNS_DIR / "model.keras",
}
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

st.set_page_config(page_title="NeuroVision", layout="wide")


@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path, compile=False)


def get_input_size(model):
    # model.input_shape = (None, H, W, C) -> cv2 wants (W, H)
    _, h, w, _ = model.input_shape
    return (w, h)


def decode_image(file_bytes):
    if not file_bytes:
        return None
    array = np.frombuffer(file_bytes, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)  # BGR, like training


def predict(model, image):
    resized = cv2.resize(image, get_input_size(model), interpolation=cv2.INTER_AREA)
    return model.predict(resized[None] / 255.0, verbose=0)[0]


def show_result(container, title, probabilities):
    best = int(np.argmax(probabilities))
    container.subheader(title)
    container.markdown(f"**{CLASS_NAMES[best]}** — {probabilities[best]:.1%}")
    for name, probability in zip(CLASS_NAMES, probabilities):
        container.progress(float(probability), text=f"{name}: {probability:.1%}")


st.title("NeuroVision")
st.write("Upload a brain MRI to predict the tumor type with 3 models.")

# load every model that exists
models = {}
for model_name, path in MODEL_PATHS.items():
    if not path.exists():
        st.error(f"Model not found: {path}")
        continue
    try:
        models[model_name] = load_model(path)
    except Exception as e:
        st.error(f"{model_name} failed to load: {e}")

if not models:
    st.stop()

uploaded_files = st.file_uploader(
    "MRI images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

for uploaded in uploaded_files:
    image = decode_image(uploaded.getvalue())
    if image is None:
        st.warning(f"{uploaded.name} could not be read.")
        continue

    all_probs = {name: predict(model, image) for name, model in models.items()}
    ensemble = np.mean(list(all_probs.values()), axis=0)

    st.divider()
    st.image(image[..., ::-1], caption=uploaded.name, width=250)  # BGR -> RGB, display only

    columns = st.columns(len(all_probs) + 1)
    for column, (name, probs) in zip(columns, all_probs.items()):
        show_result(column, name, probs)
    show_result(columns[-1], "Ensemble (average)", ensemble)