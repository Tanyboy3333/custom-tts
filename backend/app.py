from flask import Flask, request, jsonify, send_file
import os
import whisper
import torch
import pickle
from TTS.api import TTS
from tts_trainer import train_tts
from inference import generate_speech

app = Flask(__name__)

UPLOAD_FOLDER = "backend/uploads"
PROCESSED_FOLDER = "backend/processed"
MODEL_FOLDER = "backend/models"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return jsonify({"message": "Custom TTS API is running!"})

# **1️⃣ Upload Audio Files**
@app.route("/upload", methods=["POST"])
def upload_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    return jsonify({"message": "File uploaded successfully", "file_path": filepath})

# **2️⃣ Transcribe Audio**
@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    files = os.listdir(UPLOAD_FOLDER)
    transcriptions = {}
    model = whisper.load_model("base")
    
    for file in files:
        file_path = os.path.join(UPLOAD_FOLDER, file)
        result = model.transcribe(file_path)
        transcriptions[file] = result["text"]
    
    return jsonify({"transcriptions": transcriptions})

# **3️⃣ Train TTS Model**
@app.route("/train", methods=["POST"])
def train_model():
    model_path = train_tts()
    return jsonify({"message": "Model trained successfully!", "model_path": model_path})

# **4️⃣ Generate Speech**
@app.route("/tts", methods=["POST"])
def text_to_speech():
    data = request.json
    text = data.get("text", "")
    output_path = generate_speech(text)
    
    return send_file(output_path, as_attachment=True)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render's assigned port
    app.run(host="0.0.0.0", port=port)
