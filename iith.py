from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)
model = tf.keras.models.load_model('model.h5')
def preprocess_image(image, target_size):
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


        processed_image = preprocess_image(img, target_size=(512, 512))  # Adjust size based on model input size


        prediction = model.predict(processed_image)
        prediction_list = prediction.tolist()
        prediction_final = max(prediction_list)


        print()


        return jsonify({'prediction': [max(prediction_final)*100,prediction_final.index(max(prediction_final))]})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
