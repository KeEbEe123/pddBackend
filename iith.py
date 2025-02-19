from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import gdown
import os

app = Flask(__name__)
CORS(app)

# Google Drive file setup
file_id = "1ad-AerR2PjwHb4Pam95P3Cp5U-jakAsX"
model_path = "model.h5"

# Download model only if it doesn't exist
if not os.path.exists(model_path):
    print("Downloading model...")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)

# Load the model
try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Preprocessing function
def preprocess_image(image, target_size=(512, 512)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        img = Image.open(io.BytesIO(file.read()))

        # Preprocess image
        processed_image = preprocess_image(img)

        # Make prediction
        prediction = model.predict(processed_image)
        prediction_list = prediction[0].tolist()  # Extract the first row

        # Get max confidence and index
        max_confidence = max(prediction_list) * 100
        predicted_index = prediction_list.index(max(prediction_list))

        return jsonify({'confidence': max_confidence, 'class_index': predicted_index})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
