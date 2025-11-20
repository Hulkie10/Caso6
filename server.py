from flask import Flask, request, jsonify
import base64
import io
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_URL = "https://www.kaggle.com/models/kaggle/esrgan-tf2/TensorFlow2/esrgan-tf2/1"

print("Cargando modelo ESRGAN, por favor espera...")
model = hub.load(MODEL_URL)
print("Modelo cargado correctamente ✔")

def preprocess_image(img_bytes):
    img = tf.image.decode_image(img_bytes, channels=3)
    size = tf.convert_to_tensor(img.shape[:-1]) // 4 * 4
    img = tf.image.crop_to_bounding_box(img, 0, 0, size[0], size[1])
    return tf.expand_dims(tf.cast(img, tf.float32), 0)

@app.route("/upload", methods=["POST"])
def upload():
    print("→ Se recibió POST /upload")
    
    if "file" not in request.files:
        print("ERROR: no 'file' en request")
        return jsonify({"error": "No file"}), 400

    file = request.files["file"]
    img_bytes = file.read()

    hr = preprocess_image(img_bytes)
    sr = model(hr)
    sr = tf.squeeze(tf.clip_by_value(sr, 0, 255))

    img = Image.fromarray(tf.cast(sr, tf.uint8).numpy())
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")

    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    print("Base64 length:", len(img_base64))

    return jsonify({"result": img_base64})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
