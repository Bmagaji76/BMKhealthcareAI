import os
import torch
import speech_recognition as sr
from gtts import gTTS
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import requests

app = Flask(__name__)

# ✅ GitHub Release URL for the model file
GITHUB_MODEL_URL = "https://github.com/Bmagaji76/BMKhealthcareAI/releases/download/v1.0/model.safetensors"
MODEL_DIR = "conversational_medical_model"
MODEL_FILE = os.path.join(MODEL_DIR, "model.safetensors")

# ✅ Function to download the model file if not present
def download_model():
    if not os.path.exists(MODEL_FILE):
        os.makedirs(MODEL_DIR, exist_ok=True)
        print("Downloading model...")
        response = requests.get(GITHUB_MODEL_URL, stream=True)
        with open(MODEL_FILE, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print("✅ Model downloaded successfully!")

# ✅ Download the model if not already present
download_model()

# Load trained AI model
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
print("✅ Model loaded successfully!")

# Speech recognition function
def speech_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Sorry, could not understand the audio."
    except sr.RequestError:
        return "Could not request results, check your internet connection."

# Text-to-Speech function
def text_to_speech(text, filename="static/output.mp3"):
    tts = gTTS(text=text, lang='en')
    tts.save(filename)
    return filename

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
import wave
import io

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_input = request.form.get("user_input")  # Get text input
        if not user_input and "audio" not in request.files:
            return jsonify({"error": "No text or audio input received."}), 400

        # ✅ Handle audio input
        if "audio" in request.files:
            audio_file = request.files["audio"]
            audio_bytes = audio_file.read()

            # ✅ Ensure the file is a valid WAV file
            try:
                with wave.open(io.BytesIO(audio_bytes), "rb") as audio:
                    user_input = speech_to_text(io.BytesIO(audio_bytes))  # Convert speech to text
            except wave.Error:
                return jsonify({"error": "Invalid audio format. Please upload a WAV file."}), 400

        # ✅ Ensure there is text input after processing
        if not user_input:
            return jsonify({"error": "No valid input received after processing."}), 400

        # ✅ Process input with AI Model
        inputs = tokenizer(user_input, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(**inputs)
        diagnosis = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prescription = "Take necessary precautions and consult a doctor if symptoms persist."

        audio_response = text_to_speech(diagnosis)

        return jsonify({"user_input": user_input, "diagnosis": diagnosis, "prescription": prescription, "audio_response": audio_response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# API route for JSON response
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.json
    user_input = data.get("text")
    
    inputs = tokenizer(user_input, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs)
    diagnosis = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prescription = "Take necessary precautions and consult a doctor if symptoms persist."
    
    return jsonify({"diagnosis": diagnosis, "prescription": prescription})

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render assigns a port
    app.run(host="0.0.0.0", port=port, debug=True)

# Gunicorn needs 'application' instead of 'app'
application = app


