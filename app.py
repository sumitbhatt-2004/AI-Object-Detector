import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer
import av

# Load Model
model = YOLO('yolov8n.pt')

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    results = model(img, conf=0.5)
    annotated_frame = results[0].plot()
    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

st.title("AI Object Detector")
webrtc_streamer(key="example", video_frame_callback=video_frame_callback)
