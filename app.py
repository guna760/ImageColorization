import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# -----------------------------
# Load Trained Model
# -----------------------------
model = load_model("models/colorization_model.h5")

IMG_SIZE = 128

st.set_page_config(page_title="Image Colorization", layout="wide")

st.title("🎨 Black & White Image Colorization using Deep Learning")
st.write("Upload a grayscale or color image. The model will generate a colorized version.")

uploaded_file = st.file_uploader(
    "Choose an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # -----------------------------
    # Read Image
    # -----------------------------
    image = Image.open(uploaded_file).convert("RGB")

    original = np.array(image)

    # Resize according to model input
    img = cv2.resize(original, (IMG_SIZE, IMG_SIZE))

    # -----------------------------
    # Convert RGB -> LAB
    # -----------------------------
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    # Extract L channel
    L = lab[:, :, 0]

    # Normalize
    L_input = L.astype("float32") / 255.0

    # Model Input
    input_img = L_input.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    # -----------------------------
    # Prediction
    # -----------------------------
    prediction = model.predict(input_img, verbose=0)

    prediction = prediction[0]

    # Convert predicted AB values
    prediction = prediction * 128

    # -----------------------------
    # Create LAB Image
    # -----------------------------
    lab_output = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

    lab_output[:, :, 0] = L

    lab_output[:, :, 1:] = prediction

    # LAB -> RGB
    colorized = cv2.cvtColor(lab_output.astype(np.float32), cv2.COLOR_LAB2RGB)

    colorized = np.clip(colorized, 0, 1)

    colorized = (colorized * 255).astype(np.uint8)

    # -----------------------------
    # Display
    # -----------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Colorized Image")
        st.image(colorized, use_container_width=True)

    # -----------------------------
    # Download
    # -----------------------------
    output = Image.fromarray(colorized)

    st.download_button(
        label="📥 Download Colorized Image",
        data=output.tobytes(),
        file_name="colorized_image.png",
        mime="image/png"
    )