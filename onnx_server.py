from flask import Flask, request, jsonify, send_file, url_for
from flask_cors import CORS
import onnxruntime as rt
import numpy as np
import io
import zipfile
from PIL import Image
import os
import hashlib
import uuid
import time

app = Flask(__name__)
CORS(app)

# Load the ONNX model
MODEL_PATH = "generator.onnx"
session = rt.InferenceSession(MODEL_PATH)

EXPECTED_IMG_SHAPE = (128, 128, 3)
GENERATED_FILES_DIR = "generated_files"  # Directory to store generated files

# Ensure the directory exists
os.makedirs(GENERATED_FILES_DIR, exist_ok=True)

def generate_noise_vector(file_bytes=None, latent_dim=256):
    """Generate a reproducible noise vector using file bytes as a seed."""
    if file_bytes:
        file_hash = int(hashlib.sha256(file_bytes).hexdigest(), 16) % (2**32)
        rng = np.random.RandomState(file_hash)
    else:
        rng = np.random
    return rng.randn(1, latent_dim).astype(np.float32)

def run_model(noise: np.ndarray) -> Image.Image:
    """Run the ONNX model and generate an image."""
    input_feed = {
        "noise": noise,
        "alpha": np.array([1.0], dtype=np.float32)
    }
    result = session.run(None, input_feed)
    output = result[0]

    if output.ndim == 4 and output.shape[1] == 3:
        image_array = np.squeeze(output, axis=0)
        image_array = np.transpose(image_array, (1, 2, 0))
    else:
        raise ValueError(f"Unexpected output shape: {output.shape}")

    image_array = (image_array * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(image_array)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "ONNX Model API is running!"})

@app.route("/generate", methods=["POST"])
def generate():
    if 'files' not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist('files')
    if not files:
        return jsonify({"error": "No images provided"}), 400

    # Generate a unique filename for this request
    zip_filename = f"generated_{uuid.uuid4().hex}.zip"
    zip_path = os.path.join(GENERATED_FILES_DIR, zip_filename)

    # Create a ZIP archive
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file in files:
            try:
                file_bytes = file.read()
                noise = generate_noise_vector(file_bytes=file_bytes, latent_dim=256)
                generated_image = run_model(noise)

                img_byte_arr = io.BytesIO()
                generated_image.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                zip_file.writestr(f"generated_{file.filename}", img_byte_arr.getvalue())
            except Exception as e:
                print(f"Error processing {file.filename}: {e}")
                continue

    # Generate a downloadable URL
    download_url = url_for('download', filename=zip_filename, _external=True)
    return jsonify({"download_url": download_url})

@app.route("/download/<filename>", methods=["GET"])
def download(filename):
    """Serve the generated ZIP file and delete it after sending."""
    file_path = os.path.join(GENERATED_FILES_DIR, filename)

    if os.path.exists(file_path):
        # Send the file
        response = send_file(file_path, mimetype='application/zip', as_attachment=True, download_name=filename)

        # Delay deletion slightly to ensure the response is sent successfully
        time.sleep(2)
        os.remove(file_path)

        return response

    return jsonify({"error": "File not found"}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
