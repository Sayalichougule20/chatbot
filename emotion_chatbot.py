import os
import librosa
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import tensorflow as tf
import speech_recognition as sr
import nltk
import streamlit as st
import random
import webbrowser
from nltk.sentiment import SentimentIntensityAnalyzer
from tensorflow.keras.models import load_model

# Download Sentiment Intensity Analyzer (Only once)
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Load pre-trained emotion recognition model for tone analysis
model = load_model("emotion_recognition_model.h5")

# Emotion labels for tone analysis
emotion_labels = {
    0: 'neutral', 1: 'calm', 2: 'happy', 3: 'sad', 4: 'angry',
    5: 'fearful', 6: 'disgust', 7: 'surprised'
}

# Emotion-based responses & recommendations
emotion_responses = {
    "happy": ["That's awesome! 😊 Keep up the positivity!", "Happiness is contagious! Maybe try some comedy videos?"],
    "sad": ["I'm here for you. Want to talk about it? 💙", "Things will get better. Stay strong! Maybe a motivational video?"],
    "angry": ["I understand. Take a deep breath. Want some help calming down?", "Try some relaxation techniques!"],
    "neutral": ["Got it! How can I assist you today?", "Tell me more! I'm listening. 😊"]
}

video_links = {
    "happy": "https://www.youtube.com/watch?v=d-diB65scQU",
    "sad": "https://www.youtube.com/watch?v=ZXsQAXx_ao0",
    "angry": "https://www.youtube.com/watch?v=oFndzllxpLk",
    "neutral": "https://www.youtube.com/watch?v=3qHkcs3kG44"
}

# Function to extract MFCC features from audio
def extract_mfcc(file_path, max_pad_len=200):
    audio, sample_rate = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    pad_width = max_pad_len - mfccs.shape[1]
    mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant') if pad_width > 0 else mfccs[:, :max_pad_len]
    return mfccs

# Function to record audio
def record_audio(filename="user_voice.wav", duration=5, fs=44100):
    st.write("🎤 Recording... Speak now!")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.int16)
    sd.wait()
    wav.write(filename, fs, audio)
    st.success("✅ Recording saved!")

# Function to transcribe speech to text
def speech_to_text(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio)
        except (sr.UnknownValueError, sr.RequestError):
            return None

# Function to analyze sentiment from text
def analyze_sentiment(text):
    sentiment_score = sia.polarity_scores(text)
    return "happy" if sentiment_score["compound"] >= 0.05 else "sad" if sentiment_score["compound"] <= -0.05 else "neutral"

# Function to predict emotion from tone
def predict_emotion_from_tone(file_path):
    features = extract_mfcc(file_path)
    features = np.expand_dims(features, axis=[0, -1])
    prediction = model.predict(features)
    return emotion_labels[np.argmax(prediction)]

# Function to generate chatbot response
def generate_response(emotion):
    return random.choice(emotion_responses.get(emotion, ["I'm here to chat! 😊"]))

# ------------------------ 🎨 Streamlit UI with Custom Styling ------------------------

st.set_page_config(page_title="EmotiCare - Emotion Based AI Chatbot", page_icon="🎙️", layout="centered")

# Add custom styling
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #FFDEE9, #B5FFFC);
        font-family: 'Arial', sans-serif;
    }
    .title {
        color: #ff4b5c;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
    }
    .subtitle {
        color: #333;
        font-size: 18px;
        text-align: center;
    }
    .stTextInput, .stButton, .stRadio {
        border-radius: 15px !important;
    }
    .button-style {
        border-radius: 12px;
        background-color: #ff4b5c;
        color: white;
        font-size: 16px;
        padding: 10px;
        width: 100%;
        text-align: center;
        transition: 0.3s ease;
    }
    .button-style:hover {
        background-color: #ff6b7f;
    }
    .stVideo {
        border-radius: 15px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="title">🎙️ EmotiCare - Emotion Based AI Chatbot</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Detect emotions from your voice or text and receive personalized responses!</p>', unsafe_allow_html=True)

# ------------------------ User Input Selection ------------------------
st.sidebar.title("🔍 Choose Input Method")
input_type = st.sidebar.radio("", ["Text", "Voice"])

if input_type == "Text":
    user_text = st.text_input("💬 Type your message:")
    if st.button("Analyze Emotion 🎭", key="analyze_text"):
        if user_text:
            with st.spinner("🔍 Analyzing Emotion..."):
                detected_emotion = analyze_sentiment(user_text)
            st.markdown(f"### 🎭 Detected Emotion: **{detected_emotion.capitalize()}**")
            st.success(f"🤖 Chatbot: {generate_response(detected_emotion)}")
            if detected_emotion in video_links:
                st.markdown("### 📺 **Check out this video!**")
                st.video(video_links[detected_emotion])
        else:
            st.warning("⚠️ Please enter some text.")

elif input_type == "Voice":
    if st.button("🎙️ Record Voice", key="record_voice", help="Click to record your voice"):
        record_audio("user_voice.wav")

    if st.button("Analyze Voice Emotion 🎭", key="analyze_voice"):
        with st.spinner("🎧 Processing Audio..."):
            text = speech_to_text("user_voice.wav")
            detected_emotion = analyze_sentiment(text) if text else predict_emotion_from_tone("user_voice.wav")

        if text:
            st.markdown(f"### 📝 **Recorded Text:** _{text}_")  # Show recorded text

        st.markdown(f"### 🎭 Detected Emotion: **{detected_emotion.capitalize()}**")
        st.success(f"🤖 Chatbot: {generate_response(detected_emotion)}")
        if detected_emotion in video_links:
            st.markdown("### 📺 **Check out this video!**")
            st.video(video_links[detected_emotion])

# ------------------------ 🚀 Sidebar Enhancements ------------------------
st.sidebar.markdown("## 🌟 Features")
st.sidebar.write(" Detect emotion from **text** or **voice**")  
st.sidebar.write(" Get AI-generated **chatbot responses**")  
st.sidebar.write(" Receive **video recommendations** based on emotion")  
st.sidebar.write(" **Modern UI with cool animations**")  


