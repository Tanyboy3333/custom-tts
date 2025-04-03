import os

def generate_speech(text):
    output_path = "backend/output.wav"
    
    with open(output_path, "wb") as f:
        f.write(b"Fake audio file content.")  # Placeholder
    
    return output_path
