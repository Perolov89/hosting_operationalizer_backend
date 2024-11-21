from flask import Flask, request, jsonify
from flask_cors import CORS
import onnxruntime as ort
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import base64
from dotenv import load_dotenv
import os
import logging

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Serving endpoint from environment variable
serving_endpoint = os.getenv("ENDPOINT")

# Load ONNX model
try:
    onnx_session = ort.InferenceSession("model.onnx")
    logger.info("ONNX model loaded successfully.")
except Exception as e:
    logger.exception("Failed to load ONNX model.")
    raise e

# Class names corresponding to the output labels
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
               'add', 'divide', 'multiply', 'subtract']

# Initialize Flask app
app = Flask(__name__)
CORS(app)


def preprocess_image(image_bytes):
    """
    Preprocess the image for the ONNX model.

    Parameters:
        image_bytes (bytes): Raw image data in bytes.

    Returns:
        np.ndarray: Preprocessed image array ready for model input.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))

        # Convert to grayscale if not already
        if img.mode != 'L':
            img = img.convert('L')

        # Resize to match the model's input shape
        img = img.resize((28, 28))

        # Normalize pixel values and reshape for model input
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1).astype(np.float32)

        return img_array
    except UnidentifiedImageError as e:
        logger.error(f"Error processing image: {e}")
        return None
    except Exception as e:
        logger.exception("Unexpected error during image preprocessing.")
        return None


def convert_to_symbol(predicted_label):
    """
    Convert a predicted label into a human-readable symbol.

    Parameters:
        predicted_label (int): Numeric label predicted by the model.

    Returns:
        str: Symbol corresponding to the predicted label.
    """
    symbol_map = {
        'add': '+',
        'divide': '/',
        'multiply': '*',
        'subtract': '-'
    }
    predicted_class = class_names[predicted_label]
    return symbol_map.get(predicted_class, predicted_class)


@app.route('/')
def home():
    """Health check endpoint to confirm the backend is running."""
    return "The operationalizer is running!"


@app.route(serving_endpoint, methods=['POST'])
def predict():
    """
    Endpoint to process images and return model predictions.

    Expects a JSON payload with a key 'images' containing base64-encoded images
    for 'firstNumber', 'operator', and 'secondNumber'.

    Returns:
        JSON: Predictions for each image with labels, symbols, and accuracies.
    """
    try:
        data = request.get_json()

        if not data or 'images' not in data:
            logger.error("No image data provided in the request.")
            return jsonify({'error': 'No image data provided'}), 400

        required_keys = ['firstNumber', 'operator', 'secondNumber']
        if not all(key in data['images'] for key in required_keys):
            logger.error("Missing required keys in images.")
            return jsonify({'error': 'Missing required keys in images'}), 400

        predictions = {}
        for id, image_data in data['images'].items():
            try:
                # Decode base64 image data
                base64_data = image_data.split(',')[1] if ',' in image_data else image_data
                image_bytes = base64.b64decode(base64_data)

                # Preprocess the image
                img_array = preprocess_image(image_bytes)
                if img_array is None:
                    logger.error(f"Invalid image format for {id}.")
                    return jsonify({'error': f'Invalid image format for {id}'}), 400

                # Run inference on the model
                input_name = onnx_session.get_inputs()[0].name
                prediction = onnx_session.run(None, {input_name: img_array})

                # Extract prediction results
                predicted_label = np.argmax(prediction[0], axis=1)[0]
                predicted_symbol = convert_to_symbol(predicted_label)

                # Store results
                predictions[id] = {
                    'predicted_label': int(predicted_label),
                    'predicted_symbol': predicted_symbol,
                    'accuracy': float(prediction[0][0][predicted_label])
                }
            except Exception as e:
                logger.exception(f"Error processing image with ID {id}.")
                return jsonify({'error': f'Error processing image {id}: {e}'}), 500

        return jsonify(predictions)

    except Exception as e:
        logger.exception("Unexpected error in prediction endpoint.")
        return jsonify({'error': f"An unexpected error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    try:
        app.run(debug=True)
    except Exception as e:
        logger.exception("Failed to start the Flask app.")
