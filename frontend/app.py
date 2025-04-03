import streamlit as st
import requests

API_URL = "https://your-render-api-url.onrender.com"

st.title("🔊 Custom TTS Voice Trainer")

# **Upload Audio**
st.header("📤 Upload Audio Files")
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
if uploaded_file:
    files = {"file": uploaded_file.getvalue()}
    response = requests.post(f"{API_URL}/upload", files=files)
    st.write(response.json())

# **Train TTS Model**
st.header("🎤 Train TTS Model")
if st.button("Train Model"):
    response = requests.post(f"{API_URL}/train")
    st.write(response.json())

# **Chatbot for Text-to-Speech**
st.header("💬 TTS Chatbot")
text_input = st.text_area("Enter text for speech synthesis:")
if st.button("Generate Speech"):
    response = requests.post(f"{API_URL}/tts", json={"text": text_input})
    st.audio(response.content)
