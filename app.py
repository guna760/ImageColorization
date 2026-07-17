import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from io import BytesIO

# -----------------------------
# Configuration
# -----------------------------
IMG_SIZE = 128

st.set_page_config(
    page_title="Image Colorization",
    layout="wide"
)

st.title("🎨 Black & White Image Colorization")
st.write("Upload a grayscale image to generate its colorized version.")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_colorization_model():
    return load_model("models/colorization_model.h5", compile=False)

model = load_colorization_model()

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    original = np.array(image)

    # Resize for model
    resized = cv2.resize(original, (IMG_SIZE, IMG_SIZE))

    # RGB -> LAB
    lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)

    # L channel
    L = lab[:, :, 0].astype(np.float32)

    # Normalize
    L_input = L / 255.0
    L_input = L_input.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    # Prediction
    pred = model.predict(L_input, verbose=0)[0]

    # Denormalize AB channels
    pred = pred * 128

    # Create LAB image
    output_lab = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

    output_lab[:, :, 0] = L
    output_lab[:, :, 1:] = pred

    # LAB -> RGB
    colorized = cv2.cvtColor(output_lab, cv2.COLOR_LAB2RGB)

    colorized = np.clip(colorized, 0, 1)
    colorized = (colorized * 255).astype(np.uint8)

    # Display Images
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, width=500)

    with col2:
        st.subheader("Colorized Image")
        st.image(colorized, width=500)

    # Download Image
    output_image = Image.fromarray(colorized)

    buffer = BytesIO()
    output_image.save(buffer, format="PNG")
    buffer.seek(0)

    st.download_button(
        label="📥 Download Colorized Image",
        data=buffer,
        file_name="colorized_image.png",
        mime="image/png"
    )