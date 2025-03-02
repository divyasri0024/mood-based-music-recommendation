import cv2
import pandas as pd
import numpy as np
import streamlit as st
from deepface import DeepFace
from PIL import Image

# Load the dataset
df = pd.read_csv("music_dataset.csv")

# Function to detect mood from an image
def detect_mood(image):
    try:
        image = np.array(image.convert("RGB"))
        result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
        return result[0]['dominant_emotion']
    except:
        return "Error: Could not detect face."

# Function to recommend songs based on detected mood
def recommend_songs(mood):
    mood = mood.lower()
    return df[df["mood"] == mood][["song", "artist"]].values.tolist() if mood in df["mood"].values else []

# Streamlit Web App
st.title("🎵 AI Mood-Based Music Recommender")

# Upload an image
uploaded_file = st.file_uploader("Upload your image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    mood = detect_mood(image)
    
    if "Error" in mood:
        st.error(mood)
    else:
        st.success(f"Detected Mood: **{mood.capitalize()}**")
        
        songs = recommend_songs(mood)
        
        if songs:
            st.subheader("🎶 Recommended Songs for You:")
            for song, artist in songs:
                st.write(f"- **{song}** by {artist}")
        else:
            st.warning("No song recommendations available for this mood.")
