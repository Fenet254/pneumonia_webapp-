import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "model", "pneumonia_classifier_model.keras")

# Load the trained model once
model = load_model(model_path)

def predict_image(filepath):
    img = image.load_img(filepath, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]  # Binary output
    label = "Pneumonia" if prediction > 0.5 else "Normal"
    confidence = prediction if label == "Pneumonia" else 1 - prediction
    return label, float(confidence)