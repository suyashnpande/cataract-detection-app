# new file added- GPT ( for gui using streamlit)
# Install Streamlit: Install Streamlit via pip install streamlit.

# Create a Streamlit Script (app.py):
# Below is a basic template for Streamlit to create a simple interface for your model:

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
@st.cache_resource
def load_cataract_model():
    return load_model('final_model.keras')

model = load_cataract_model()

# Preprocess the image
def preprocess_image(img):
    img = img.resize((224, 224))  # Ensure the image size matches the model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize to [0, 1]
    return img_array

# Prediction function
def predict_cataract(img):
    img_array = preprocess_image(img)
    prediction_prob = model.predict(img_array)[0][0]
    is_cataract = prediction_prob <= 0.5
    confidence = round(1 - prediction_prob if is_cataract else prediction_prob, 3)
    return bool(is_cataract), confidence

# Streamlit UI
st.title("Cataract Detection App")
st.write("Upload an eye image to detect the presence of cataracts.")

uploaded_file = st.file_uploader("Choose an eye image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    # Make prediction
    has_cataract, confidence = predict_cataract(img)

    # Display the result
    if has_cataract:
        st.error(f"Cataract detected with {confidence*100}% confidence.")
    else:
        st.success(f"No cataract detected with {confidence*100}% confidence.")
