from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "mri_cnn.keras"
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_SIZE = (224, 224)

st.set_page_config(page_title="NeuroVision")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def prepare_image(file_bytes):
    if not file_bytes:
        return None
    array = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)   # stays BGR, like training


st.title("NeuroVision")
st.write("Upload a brain MRI to predict the tumor type.")

if not MODEL_PATH.exists():
    st.error(f"Model not found: {MODEL_PATH}")
    st.stop()

model = load_model()
uploaded_files = st.file_uploader(
    "MRI images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)


for uploaded in uploaded_files:
    image = prepare_image(uploaded.getvalue())
    if image is None:
        st.warning(f"{uploaded.name} could not be read.")
        continue

    probabilities = model.predict(image[None] / 255.0, verbose=0)[0]
    best = int(np.argmax(probabilities))

    st.divider()
    left, right = st.columns(2)
    left.image(image[..., ::-1], caption=uploaded.name, width=250)   # BGR -> RGB, display only
    right.subheader(CLASS_NAMES[best])
    right.write(f"Confidence: {probabilities[best]:.1%}")
    for name, probability in zip(CLASS_NAMES, probabilities):
        right.progress(float(probability), text=f"{name}: {probability:.1%}")