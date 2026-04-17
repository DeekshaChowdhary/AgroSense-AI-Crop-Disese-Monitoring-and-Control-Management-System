from flask import Flask, render_template, request, jsonify, send_from_directory
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import json
import os
import uuid

app = Flask(__name__)

# Paths
MODEL_PATH = 'model/model.keras'
CLASS_NAMES_PATH = 'model/class_names.json'
DISEASE_INFO_PATH = 'model/disease_info.json'
UPLOAD_FOLDER = 'uploads'

# Create uploads folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and data
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = json.load(f)

with open(DISEASE_INFO_PATH, 'r') as f:
    disease_info = json.load(f)

IMG_SIZE = (160, 160)


# Prediction function
def predict_disease(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)
    class_idx = np.argmax(preds[0])
    confidence = float(preds[0][class_idx]) * 100

    class_name = class_names[class_idx]
    info = disease_info.get(class_name, {})

    treatment = info.get('treatment', {})
    prevention = info.get('prevention', {})

    # Ensure dictionary format
    if isinstance(treatment, str):
        treatment = {"Info": treatment}
    if isinstance(prevention, str):
        prevention = {"Info": prevention}

    return {
        'disease': class_name,
        'confidence': f"{confidence:.2f}%",
        'severity': info.get('severity', 'Unknown'),
        'treatment': treatment,
        'prevention': prevention
    }


# Routes
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save file
    ext = file.filename.split('.')[-1]
    unique_name = f"{uuid.uuid4()}.{ext}"
    upload_path = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(upload_path)

    # Predict
    result = predict_disease(upload_path)
    result['image_path'] = f"/uploads/{unique_name}"

    return jsonify(result)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# Run app
if __name__ == '__main__':
    app.run(debug=True)
