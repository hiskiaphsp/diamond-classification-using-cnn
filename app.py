from flask import Flask, render_template, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the MobileNetV2 model
model = load_model('mobilenetv2_model.h5')  # Ganti dengan path aktual ke model MobileNetV2 Anda

# Function to load and preprocess a single image for MobileNetV2
def preprocess_single_image_mobilenetv2(image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Create the 'uploads' directory if it doesn't exist
        if not os.path.exists('static/uploads'):
            os.makedirs('static/uploads')

        # Gunakan secure_filename untuk memastikan nama file aman
        filename = secure_filename(file.filename)
        file_path = os.path.join('static/uploads', filename)
        file.save(file_path)

        # Preprocess the image for MobileNetV2
        preprocessed_image = preprocess_single_image_mobilenetv2(file_path)

        # Make predictions
        predictions = model.predict(preprocessed_image)

        # Extract the predicted class label and probability directly
        class_indices = {0: 'cushion', 1: 'emerald', 2: 'heart', 3: 'marquise', 4: 'oval', 5: 'pear', 6: 'princess', 7: 'round'}  # Update with your actual class indices
        predicted_class_index = np.argmax(predictions)
        predicted_class_label = class_indices[predicted_class_index]
        probability = predictions[0][predicted_class_index]

        # Return the predicted class label, probability, and image filename
        return jsonify({
            'predicted_class': predicted_class_label,
            'probability': float(probability),
            'image_filename': filename
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
