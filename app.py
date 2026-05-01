import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av # Isse video frames handle hote hain link par

# Page Config
st.set_page_config(page_title="AI Object Detector", layout="wide")
st.title("🚀 Real-Time Object Detection (YOLOv8)")

# Load Model
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# Logic for processing video frames from Browser
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24") # Browser se frame lena

        # YOLO Inference
        results = model.predict(img, conf=0.5)
        
        # Annotate (Box banana)
        annotated_frame = results[0].plot()

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# UI - Yahan se camera start hoga
st.write("Niche 'Start' button par click karein aur camera allow karein.")

webrtc_streamer(
    key="object-detection",
    video_processor_factory=VideoProcessor,
    rtc_configuration={ # Ye part cloud par camera connectivity ke liye zaroori hai
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)

