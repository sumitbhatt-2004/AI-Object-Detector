import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np

# Page Config
st.set_page_config(page_title="AI Object Detector", layout="wide")
st.title("🚀 Real-Time Object Detection (YOLOv8)")
st.write("Deep Learning model use karke objects pehchane!")

# Load Model
@st.cache_resource # Isse model baar baar load nahi hoga (Performance boost)
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# Sidebar Settings
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

# Camera Logic
run = st.checkbox('Start Webcam')
FRAME_WINDOW = st.image([]) # Khali jagah jahan video dikhegi

camera = cv2.VideoCapture(0)

while run:
    _, frame = camera.read()
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Streamlit RGB use karta hai
        
        # Inference
        results = model.predict(frame, conf=conf_threshold)
        
        # Plot results
        annotated_frame = results[0].plot()
        
        # Display in Streamlit app
        FRAME_WINDOW.image(annotated_frame)
    else:
        st.error("Camera nahi mil raha. Check permissions!")
        break
else:
    st.write('Webcam Stopped')
    camera.release()
