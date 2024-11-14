from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Load your trained model
model = load_model("mnist_digit_recognition_model.keras")
model.save("mnist_digit_recognition_model.keras")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    # Get the image from the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    # Open the image and preprocess it
    img = Image.open(file).convert('L').resize((28, 28))  # Resize and convert to grayscale
    img = np.array(img) / 255.0  # Normalize pixel values
    img = img.reshape(1, 28, 28)  # Reshape for model input

    # Make prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]

    return jsonify({"predicted_class": int(predicted_class)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
