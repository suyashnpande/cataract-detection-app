# from flask import Flask, request, jsonify
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np

# app = Flask(__name__)

# # Load the pre-trained model
# model = load_model('final_model.keras')

# def preprocess_image(img_path):
#     """Load and preprocess the image."""
#     img = image.load_img(img_path, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0) 
#     img_array /= 255.0  # Normalize to [0, 1]
#     return img_array

# @app.route('/predict', methods=['POST'])
# def predict():
#     """Predict whether an image has cataracts or not."""
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file provided'}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     try:
#         img_path = 'temp.png'  
#         file.save(img_path)
#         img_array = preprocess_image(img_path)
        
#         # Predict
#         prediction_prob = model.predict(img_array)[0][0]
#         is_cataract = prediction_prob <= 0.5
#         if is_cataract:
#             confidence = round(1 - prediction_prob, 3)  # Confidence for cataract
#         else:
#             confidence = round(prediction_prob, 3)  
#         result = {'has_cataract': bool(is_cataract), 'confidence': float(confidence)}
        
#         return jsonify(result)

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)




# import streamlit as st
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np

# # Load the pre-trained model
# model = load_model('final_model.keras')

# def preprocess_image(img_path):
#     """Load and preprocess the image."""
#     img = image.load_img(img_path, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0  # Normalize to [0, 1]
#     return img_array

# # Streamlit App
# st.title("Cataract Detection App")
# st.write("Upload an eye image to detect cataracts.")

# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

# if uploaded_file is not None:
#     # Save the uploaded file temporarily
#     img_path = 'temp.png'
#     with open(img_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())

#     # Preprocess the image
#     img_array = preprocess_image(img_path)

#     # Predict
#     prediction_prob = model.predict(img_array)[0][0]
#     is_cataract = prediction_prob <= 0.5
#     confidence = round(1 - prediction_prob, 3) if is_cataract else round(prediction_prob, 3)

#     result = 'Cataract detected' if is_cataract else 'No cataract detected'
#     st.success(result)
#     st.write(f"Confidence: {confidence}")






# =====
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image  # Renamed to avoid conflicts
import numpy as np
from PIL import Image

# Load the pre-trained model
model = load_model('final_model.keras')

def preprocess_image(img):
    """Load and preprocess the image."""
    img = img.resize((224, 224))  # Resize the image
    img_array = keras_image.img_to_array(img)  # Use keras_image to avoid conflict
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize to [0, 1]
    return img_array

def categorize_risk(prediction_prob):
    """Categorize the risk level based on prediction probability."""
    if prediction_prob <= 0.3:
        return "High Risk"
    elif prediction_prob <= 0.5:
        return "Medium Risk"
    else:
        return "Low Risk"

# Streamlit App
st.title("👁️ Cataract Detection App")
st.write("Upload an eye image to detect cataracts. This app will classify the cataract risk level based on the uploaded image.")

uploaded_file = st.file_uploader("Choose an eye image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Show the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    img_array = preprocess_image(image)

    # Predict
    prediction_prob = model.predict(img_array)[0][0]
    is_cataract = prediction_prob <= 0.5
    risk_level = categorize_risk(prediction_prob)
    
    # Create columns for results display
    col1, col2 = st.columns(2)
    with col1:
        if is_cataract:
            result = '🚨 Cataract detected'
        else:
            result = '✅ No cataract detected'
        
        st.markdown(f"<h3 style='text-align: center; color: {'red' if is_cataract else 'green'};'>{result}</h3>", unsafe_allow_html=True)
        st.write(f"**Confidence Level:** {round(prediction_prob * 100, 2)}%")
        st.write(f"**Risk Level:** {risk_level}")

    with col2:
        st.subheader("Understanding Cataracts")
        st.write("""
    Cataracts are a clouding of the lens in the eye, which can lead to decreased vision. Here are some key points about cataracts:

    - **What are Cataracts?**: Cataracts occur when the proteins in the lens of the eye clump together, causing cloudy vision.
    
    - **Symptoms**:
        - Blurry or cloudy vision
        - Difficulty seeing at night
        - Sensitivity to light and glare
        - Seeing "halos" around lights
        - Frequent changes in prescription glasses or contact lenses

    - **Causes**:
        - Aging
        - Prolonged exposure to sunlight (UV rays)
        - Smoking and alcohol use
        - Certain medical conditions, such as diabetes
        - Family history of cataracts

    - **Treatment**: 
        - Cataracts can be treated effectively with surgery. During this procedure, the cloudy lens is removed and replaced with an artificial lens.

    - **Prevention Tips**:
        - Regular eye exams to detect cataracts early
        - Wearing sunglasses that block UV rays
        - Maintaining a healthy diet rich in vitamins C and E
        - Quitting smoking and reducing alcohol consumption
""")


# Add footer with a message
st.markdown("---")
st.markdown("<h6 style='text-align: center;'>Cataract Detection App </h6>", unsafe_allow_html=True)
