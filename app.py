from flask import Flask, request, jsonify, send_file
from io import BytesIO
from PIL import Image, ImageOps
import numpy as np
from keras.models import load_model

# Load your model and class names
model = load_model("./keras_model.h5", compile=False)
class_names = open("./labels.txt", "r").readlines()

# Initialize Flask app
app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "Server is running!", 200  # Return a simple message

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']

    if file.filename == '' or not file:
        return jsonify({"error": "File name is empty. Please upload a valid image file."}), 400

    details = {
        "general": {
            "imageUrl": "general",
            "description": "general",
        },
        "foodwaste": {
            "imageUrl": "foodwaste",
            "description": "foodwaste",
        },
        "hazardous": {
            "imageUrl": "hazardous",
            "description": "hazardous",
        },
        "recycle": {
            "imageUrl": "recycle",
            "description": "recycle",
        },
    }

    img = Image.open(file)
    img = img.convert("RGB")
    img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.asarray(img)
    normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1
    data = np.expand_dims(normalized_image_array, axis=0)

    prediction = model.predict(data)
    predicted_class = class_names[np.argmax(prediction[0])].strip()
    predictedType = predicted_class.split(" ")[1]

    return jsonify({
        "type": predictedType,
        "imageUrl": details[predictedType]["imageUrl"],
        "description": details[predictedType]["description"],
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
