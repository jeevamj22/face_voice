<<<<<<< HEAD
import cv2
import numpy as np
import mediapipe as mp
import streamlit as st

# Key points using MP holistics
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Initialize holistic model
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to perform mediapipe detection
def mediapipe_detection(image, model):
    # Convert the image to RGB format
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the RGB image
    results = model.process(rgb_image)
    
    # Convert the RGB image back to BGR format
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    
    return bgr_image, results

# Function to check if the mouth is open
def is_mouth_open(results):
    if results.face_landmarks and results.face_landmarks.landmark:
        upper_lip_index = 13
        lower_lip_index = 14
        upper_lip_y = results.face_landmarks.landmark[upper_lip_index].y
        lower_lip_y = results.face_landmarks.landmark[lower_lip_index].y
        lip_distance = lower_lip_y - upper_lip_y
        mouth_open_threshold = 0.02
        is_open = lip_distance > mouth_open_threshold
        return is_open
    else:
        return False

# Function to update the mouth status label
def update_mouth_status(is_open):
    status_text = "Mouth is open" if is_open else "Mouth is closed"
    st.text("Mouth Status: " + status_text)

# Open the webcam
cap = cv2.VideoCapture(0)

# Start the Streamlit app
st.title("Mouth Open Detection")

# Infinite loop to continuously update the video feed
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    # Perform Mediapipe detection
    image, results = mediapipe_detection(frame, holistic)

    # Check if the mouth is open
    mouth_open = is_mouth_open(results)

    # Update the mouth status label
    update_mouth_status(mouth_open)

    # Display the video feed in the Streamlit app
    st.image(image, channels="BGR", use_column_width=True)

# Release the webcam capture (this part will not be reached in this setup)
cap.release()
=======
import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from streamlit_option_menu import option_menu

# Key points using MP holistics
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Initialize holistic model
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to perform mediapipe detection
def mediapipe_detection(image, model):
    # Convert the image to RGB format
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the RGB image
    results = model.process(rgb_image)
    
    # Convert the RGB image back to BGR format
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    
    return bgr_image, results

# Function to check if the mouth is open
def is_mouth_open(results):
    if results.face_landmarks and results.face_landmarks.landmark:
        upper_lip_index = 13
        lower_lip_index = 14
        upper_lip_y = results.face_landmarks.landmark[upper_lip_index].y
        lower_lip_y = results.face_landmarks.landmark[lower_lip_index].y
        lip_distance = lower_lip_y - upper_lip_y
        mouth_open_threshold = 0.02
        is_open = lip_distance > mouth_open_threshold
        return is_open
    else:
        return False

# Function to update the mouth status label
def update_mouth_status(is_open):
    status_text = "Mouth is open" if is_open else "Mouth is closed"
    st.text("Mouth Status: " + status_text)

# Open the webcam
cap = cv2.VideoCapture(0)

# Start the Streamlit app
st.title("Mouth Open Detection")

# Infinite loop to continuously update the video feed
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    # Perform Mediapipe detection
    image, results = mediapipe_detection(frame, holistic)

    # Check if the mouth is open
    mouth_open = is_mouth_open(results)

    # Update the mouth status label
    update_mouth_status(mouth_open)

    # Display the video feed in the Streamlit app
    st.image(image, channels="BGR", use_column_width=True)

# Release the webcam capture (this part will not be reached in this setup)
cap.release()
>>>>>>> 1c816771fd78f13bc2afe97a5076a46f5a1afa77
