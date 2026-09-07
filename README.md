# Real-Time Fall Detection System

Detect human falls and falling actions in real time using videos and custom object detection.

## Dataset & Model

* **Model:** [YOLOv8n (YOLOv8 nano)](https://platform.ultralytics.com/ultralytics/yolov8/yolov8n) fine-tuned for core action and posture localization.
* **Dataset:** [Human Fall Detection Dataset](https://universe.roboflow.com/humanfalldetection/human-fall-e2evv) managed via Roboflow for training and validating model weights.

## How It Works

* **Frame Capture:** OpenCV reads an input video file (`.mp4`, `.avi`, or `.mov`) frame by frame from start to finish.
* **Fall Detection:** A fine-tuned YOLOv8 nano model scans each frame to locate individuals and categorizes their posture into specific classes (**Fallen** or **Falling**).
* **Confidence Filtering:** A dynamic slider allows users to adjust the detection confidence threshold to control sensitivity.
* **Video Annotation:** The script draws colored bounding boxes around the subjects, labels them with their corresponding class (**Fallen** / **Falling**) and confidence score, and compiles the processed frames.
* **Streamlit Delivery:** A responsive web interface displays progress tracking and renders the fully annotated, playable video directly in the browser.

## Tech Stack

* **Object Detection:** YOLOv8 (Ultralytics) fine-tuned on custom fall detection data
* **Web Framework:** Streamlit
* **Video & Image Processing:** OpenCV (`cv2`) & NumPy
* **Environment:** Google Colab with Google Drive storage

## How to Run the Project

1. Click the link to visit the webapp hosted on Streamlit.
2. Upload a video file containing human movement and view the processed fall detection output!

## Demo

https://github.com/user-attachments/assets/84f1259c-ee6c-4067-a602-2bda8937c6a1

