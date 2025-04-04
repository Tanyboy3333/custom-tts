from flask import Flask, request, jsonify, send_file
import os

app = Flask(__name__)

# ✅ Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
PROCESSED_FOLDER = os.path.join(BASE_DIR, "processed")
MODEL_FOLDER = os.path.join(BASE_DIR, "models")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# ✅ Home Route
@app.route("/")
def home():
    return jsonify({"message": "Custom TTS API is running!"})

# ✅ Upload Audio
@app.route("/upload", methods=["POST"])
def upload_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    return jsonify({"message": "File uploaded successfully", "file_path": filepath})

# ✅ Transcribe Audio (Lazy Load Whisper)
@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    try:
        files = os.listdir(UPLOAD_FOLDER)
        if not files:
            return jsonify({"error": "No files found for transcription"}), 400

        import whisper  # Lazy load Whisper
        model = whisper.load_model("tiny")  # Use 'tiny' model for speed

        transcriptions = {}
        for file in files:
            file_path = os.path.join(UPLOAD_FOLDER, file)
            result = model.transcribe(file_path)
            transcriptions[file] = result["text"]

        return jsonify({"transcriptions": transcriptions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Train TTS Model (Lazy Load)
@app.route("/train", methods=["POST"])
def train_model():
    try:
        from tts_trainer import train_tts
        model_path = train_tts()
        return jsonify({"message": "Model trained successfully!", "model_path": model_path})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Text to Speech (Lazy Load)
@app.route("/tts", methods=["POST"])
def text_to_speech():
    try:
        data = request.json
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "No text provided for TTS"}), 400

        from inference import generate_speech
        output_path = generate_speech(text)

        return send_file(output_path, as_attachment=True)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
