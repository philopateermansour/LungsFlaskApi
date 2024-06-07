import cv2
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
from PIL import Image
from transformers import AutoFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Load the first model - Classfication -
loaded_model = load_model('model.h5')

# Load the second model and tokenizer   -- Genrate Report --
encoder_checkpoint = "google/vit-base-patch16-224-in21k"
decoder_checkpoint = "ahmedabdo/facebook-bart-base-finetuned"
feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(decoder_checkpoint)
model = VisionEncoderDecoderModel.from_pretrained(decoder_checkpoint).to('cpu')

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (224, 224))
    input_image = np.expand_dims(resized_image, axis=0)
    return input_image

def predict_image(image_path):
    input_image = preprocess_image(image_path)
    predictions = loaded_model.predict(input_image)
    return predictions

def get_predicted_label(predictions):
    int_to_label = {0: 'Arm', 1: 'Chest', 2: 'Knee', 3: 'Vertebrae'}
    predicted_label = int_to_label[np.argmax(predictions)]
    return predicted_label

def predict_second_model(image):
    features = feature_extractor(image, return_tensors="pt").pixel_values.to("cpu")
    caption = tokenizer.decode(model.generate(features, max_length=1024)[0], skip_special_tokens=True)
    return caption

# Route to predict image label using the first model
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file. Please upload a chest X-ray image.'})
    image_path = file.filename
    file.save(image_path)
    predictions = predict_image(image_path)
    predicted_label = get_predicted_label(predictions)
    
    if predicted_label == 'Chest':
        img = Image.open(image_path).convert("RGB")
        caption = predict_second_model(img)
        return jsonify({'predicted_label': predicted_label,'caption': caption})
    else:
        return jsonify({'predicted_label': predicted_label, 'caption': 'Please upload a chest X-ray image.'})
