import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
import gtts
from playsound import playsound
import speech_recognition as sr
import streamlit as st
import numpy as np
import os

# Load the pre-trained sign language model
model = load_model('C:/Users/KIIT/ML Python/Accessibility Assistant/prediction.h5')

# Function to draw landmarks on the image
def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=2), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=1, circle_radius=2)
                             ) 
     #Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(13, 13, 14), thickness=1, circle_radius=3), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=1, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(13, 13, 14), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(13, 13, 14), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 

mp_holistic = mp.solutions.holistic 

# Function to perform Mediapipe detection
def mediapipe_detect(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image) #Prediction
    image.flags.writeable = True  #Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

# Function to extract keypoints from results
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# Function for sign language recognition
def sign_language_recognition():
    """
    Performs sign language recognition using the loaded model and displays the result.
    Displays the webcam feed and the recognized action or sentence.
    """

    # Initialize variables
    sequence = []
    predictions = []
    threshold = 0.8
    actions = ['hello', 'thanks', 'please', 'I love you', 'goodbye'] 
    current_action = ""

    # Start video capture
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read frame and perform detections
            ret, frame = cap.read()
            image, results = mediapipe_detect(frame, holistic)

            # Extract keypoints
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            # Predict if there's a sequence of 30 frames
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predicted_action = actions[np.argmax(res)]

                # Update predictions list and current action only if confident and different
                if res[np.argmax(res)] > threshold and predicted_action != current_action:
                    predictions.append(predicted_action)
                    current_action = predicted_action

            # Display webcam feed and recognized action/sentence using OpenCV
            cv2.putText(image, current_action, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('Sign Language Recognition', image)

            # Exit loop if 'q' key is pressed
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Function for text to speech
def text_to_speech(text):
    """
    Converts text to speech using gTTS and plays the audio.
    """

    try:
        # Create a custom directory (if it doesn't exist)
        audio_dir = "my_audio_files"  # Replace with your desired directory name
        os.makedirs(audio_dir, exist_ok=True)  # Create directory if it doesn't exist

        # Create a unique filename using UUID
        import uuid
        filename = os.path.join(audio_dir, f"temp_audio_{uuid.uuid4()}.mp3")

        # Save the generated audio to the temporary file
        with open(filename, "wb") as f:
            tts = gtts.gTTS(text)
            tts.write_to_fp(f)

        # Play the audio file
        playsound(filename)

        # Remove the temporary file (optional)
        os.remove(filename)  # Uncomment if you don't want to keep the files

    except Exception as e:
        print("Error occurred during text-to-speech conversion:", e)

# Function for speech to text
def speech_to_text():
    """
    Converts speech input by the user to text.
    Listens for microphone input for 3 seconds and displays the recognized text.
    """

    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Loop for continuous listening (limited to 3 seconds per iteration)
    while True:
        print("Listening...")

        # Start listening with timeout
        with sr.Microphone() as source:
            # Adjust ambient noise level for better recognition
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=3)

        try:
            # Try recognizing speech
            text = recognizer.recognize_google(audio)
            print("You said:", text)
            return text  # Return recognized text and exit the function
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
