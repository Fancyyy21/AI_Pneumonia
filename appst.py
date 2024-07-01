import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the trained model
model = load_model('chest_xray.h5')

# Define the path to save uploaded images
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def load_and_preprocess_image(file):
    try:
        img = image.load_img(file, target_size=(150, 150))  # Adjust target size if necessary
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x /= 255.0  # Normalize the image data
        return x
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

def predict_image(file):
    try:
        x = load_and_preprocess_image(file)
        if x is not None:
            classes = model.predict(x)
            result = np.argmax(classes, axis=-1)[0]

            # Interpret the result
            if result == 0:
                prediction = "Result is Normal"
            else:
                prediction = "Person is Affected By PNEUMONIA"
            return prediction
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# Streamlit app
st.title("Chest X-ray Pneumonia Detection")

st.write("Upload a chest X-ray image to predict if the person is affected by pneumonia.")

file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if file is not None:
    file_path = os.path.join(UPLOAD_FOLDER, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    st.image(file_path, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    prediction = predict_image(file_path)
    if prediction:
        st.write(prediction)
