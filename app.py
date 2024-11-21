from flask import Flask, request, jsonify
from flask_cors import CORS
import onnxruntime as ort
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import base64
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Serving endpoint from environment variable
serving_endpoint = os.getenv("ENDPOINT")

# Load ONNX model
onnx_session = ort.InferenceSession("model.onnx")

# Class names corresponding to the output labels
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
               'add', 'divide', 'multiply', 'subtract']

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Preprocess image


def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))

        # Convert to grayscale if not already
        if img.mode != 'L':
            img = img.convert('L')

        # Resize to match the model's input shape
        img = img.resize((28, 28))

        # Normalize pixel values to range [0, 1]
        img_array = np.array(img) / 255.0

        # Reshape to match the model input: (batch_size, height, width, channels)
        img_array = img_array.reshape(1, 28, 28, 1).astype(np.float32)

        # Save the processed image for debugging purposes
        Image.fromarray(
            (img_array[0, :, :, 0] * 255).astype(np.uint8)).save('processed_image.png')

        return img_array
    except UnidentifiedImageError as e:
        print(f"Error processing image: {e}")
        return None

# Convert label to symbol


def convert_to_symbol(predicted_label):
    symbol_map = {
        'add': '+',
        'divide': '/',
        'multiply': '*',
        'subtract': '-'
    }

    # Get the class name for the predicted label
    predicted_class = class_names[predicted_label]

    # Convert to symbol if it's an operator, otherwise return the number
    return symbol_map.get(predicted_class, predicted_class)


@app.route('/')
def home():
    return "The operationalizer is running!"


@app.route(serving_endpoint, methods=['POST'])
def predict():
    data = request.get_json()

    if not data or 'images' not in data:
        return jsonify({'error': 'No image data provided'}), 400

    required_keys = ['firstNumber', 'operator', 'secondNumber']
    if not all(key in data['images'] for key in required_keys):
        return jsonify({'error': 'Missing required keys in images'}), 400

    try:
        predictions = {}
        for id, image_data in data['images'].items():
            # Extract base64-encoded string and decode
            base64_data = image_data.split(
                ',')[1] if ',' in image_data else image_data
            image_bytes = base64.b64decode(base64_data)

            # Preprocess the image
            img_array = preprocess_image(image_bytes)
            if img_array is None:
                return jsonify({'error': f'Invalid image format for {id}'}), 400

            # Prepare input for ONNX Runtime
            input_name = onnx_session.get_inputs()[0].name
            prediction = onnx_session.run(None, {input_name: img_array})

            # Get the predicted label and symbol
            predicted_label = np.argmax(prediction[0], axis=1)[0]
            predicted_symbol = convert_to_symbol(predicted_label)

            # Return predictions with accuracy
            predictions[id] = {
                'predicted_label': int(predicted_label),
                'predicted_symbol': predicted_symbol,
                'accuracy': float(prediction[0][0][predicted_label])
            }
        return jsonify(predictions)

    except (ValueError, base64.binascii.Error) as e:
        return jsonify({'error': f'There was an error processing one or more images: {e}'}), 400


if __name__ == '__main__':
    app.run(debug=True)
