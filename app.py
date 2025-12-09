import streamlit as st
import sys
print("Python:", sys.version)
print("Installed packages:")
import pkg_resources
print([p.key for p in pkg_resources.working_set])



import cv2
from ultralytics import YOLO

st.set_page_config(page_title="Live Object Detection")

st.title("📸 Live Object Detection (Front / Back Camera)")

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Camera selection
camera_id = st.selectbox(
    "Select Camera",
    options=[0, 1, 2],
    format_func=lambda x: f"Camera {x}"
)

run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

cap = None

if run:
    cap = cv2.VideoCapture(camera_id)

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("⚠️ Unable to access the selected camera.")
        break

    # Object detection
    results = model(frame)
    annotated = results[0].plot()

    # Convert BGR → RGB
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # Show in streamlit
    FRAME_WINDOW.image(annotated)

if cap:
    cap.release()