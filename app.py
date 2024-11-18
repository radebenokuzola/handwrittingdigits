from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from io import BytesIO

app = Flask(__name__)

# URL of the externally hosted model file
model_url = "https://drive.google.com/uc?export=download&id=1W1_5X5NwIoeXu2F7XgQdu-qqbz0o5GVz"

# Download and load the model dynamically
response = requests.get(model_url)
model = tf.keras.models.load_model(BytesIO(response.content))

def preprocess_image(image):
    image = image.resize((28, 28))
    image = np.array(image.convert('L'))
    image = image / 255.0  # Normalize
    image = image.reshape(1, 28, 28, 1)  # Reshape for model input
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    image = Image.open(file.stream)
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction)
    return jsonify({'prediction': int(predicted_digit)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
