import tempfile
import cv2
import streamlit as st
import numpy as np
from ultralytics import YOLO

st.set_page_config(page_title="Fall Detection System", page_icon="🚨", layout="centered")

st.title("🚨 Fall Detection Video Processor")
st.write("Upload a video to process it frame-by-frame and view the fully annotated output with bounding boxes, class names, and confidence scores.")

@st.cache_resource
def load_model():
    return YOLO("fall_detection_best.pt")

model = load_model()

conf_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps:
        fps = 30
        
    out_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(out_file.name, fourcc, fps, (width, height))
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0
    
    status_text.text("Processing video with YOLOv8...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model(frame, conf=conf_threshold)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)
        
        current_frame += 1
        if total_frames > 0:
            progress_bar.progress(min(current_frame / total_frames, 1.0))
            
    cap.release()
    out.release()
    
    status_text.text("Processing complete! Displaying annotated video:")
    progress_bar.empty()
    
    video_bytes = open(out_file.name, 'rb').read()
    st.video(video_bytes)