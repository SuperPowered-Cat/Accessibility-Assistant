import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
import gtts
from playsound import playsound
import speech_recognition as sr
import streamlit as st
import numpy as np
import os
import functions.py

model = load_model('Models/prediction.h5)

def main():
    st.title("Accessibility Assistant")
    st.write("By Om Thakur")
    st.write("This Accessibility Assistant helps bridge communication gaps. It requires access to your webcam and microphone for functionalities like sign language recognition and speech-to-text conversion. Created using Tensorflow, Mediapipe & Keras libraries")

    st.header("Functionalities")

    if st.button("Text to Speech"):
        user_text = st.text_input("Enter text to convert to speech:")
        if user_text:
            functions.text_to_speech(user_text)

    if st.button("Sign Language"):
        functions.sign_language_recognition()

    if st.button("Speech to Text"):
        st.write("Speak now...")
        result = functions.speech_to_text()
        st.write(f"You said: {result}.")

if __name__ == "__main__":
    main()
