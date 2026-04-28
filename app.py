import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="MotorBike Helmet Violation Detection System",
    page_icon="🪖",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* Background */
.stApp {
    background-color: #0f172a;
}

/* Title */
.main-title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    color: #38bdf8;
    margin-bottom: 5px;
}

/* Subtitle */
.sub-text {
    text-align: center;
    color: #94a3b8;
    font-size: 16px;
    margin-bottom: 25px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827;
    padding: 20px;
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

/* Sidebar title */
.sidebar-title {
    font-size: 20px;
    font-weight: bold;
    color: #38bdf8;
    margin-bottom: 10px;
}

/* Card style */
.card {
    background: #1f2937;
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 15px;
}

/* File uploader */
.stFileUploader {
    background-color: #1f2937;
    padding: 10px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">MotorBike Helmet Violation Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">AI-based Helmet Detection using YOLO</div>', unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    model.to("cpu")
    return model

model = load_model()

# ---------------- SIDEBAR ----------------
st.sidebar.markdown("<div class='sidebar-title'>⚙️ Control Panel</div>", unsafe_allow_html=True)

input_type = st.sidebar.radio(
    "📌 Select Input Type",
    ["Image", "Video", "Webcam"]
)

confidence = st.sidebar.slider(
    "🎯 Confidence Threshold (%)",
    min_value=10,
    max_value=100,
    value=45
) / 100.0

st.sidebar.markdown("---")

st.sidebar.markdown("""
<div class='card'>
<strong>🟢 Green:</strong> With Helmet <br>
<strong>🔴 Red:</strong> Without Helmet
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class='card'>
📌 Tip: Use clear motorbike images/videos for best accuracy.
</div>
""", unsafe_allow_html=True)

# ---------------- DRAW FUNCTION ----------------
def draw_boxes(frame, results, threshold):
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < threshold:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label_name = results.names[cls]

        label = f"{label_name} {int(conf * 100)}%"

        # Color logic
        if "without" in label_name.lower():
            color = (0, 0, 255)  # Red
        else:
            color = (0, 255, 0)  # Green

        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # Big readable text
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            3
        )

    return frame

# ---------------- IMAGE ----------------
if input_type == "Image":

    st.markdown("""
    <h2 style='text-align:center; color:#38bdf8;'>📷 Image Detection</h2>
    <p style='text-align:center; color:#94a3b8;'>
    ⚠️ Upload a clear image with motorbike and rider for best accuracy
    </p>
    """, unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        results = model(img)[0]
        img = draw_boxes(img, results, confidence)

        st.image(img, channels="BGR", use_container_width=True)

# ---------------- VIDEO ----------------
elif input_type == "Video":

    st.markdown("""
    <h2 style='text-align:center; color:#38bdf8;'>🎥 Video Detection</h2>
    <p style='text-align:center; color:#94a3b8;'>
    Upload a video containing motorbike scenes
    </p>
    """, unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)[0]
            frame = draw_boxes(frame, results, confidence)

            stframe.image(frame, channels="BGR", use_container_width=True)

        cap.release()

# ---------------- WEBCAM ----------------
elif input_type == "Webcam":

    st.markdown("""
    <h2 style='text-align:center; color:#38bdf8;'>📸 Webcam Detection</h2>
    <p style='text-align:center; color:#94a3b8;'>
    Live detection using your camera
    </p>
    """, unsafe_allow_html=True)

    run = st.checkbox("Start Webcam")

    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            st.error("Webcam not accessible")
            break

        results = model(frame)[0]
        frame = draw_boxes(frame, results, confidence)

        FRAME_WINDOW.image(frame, channels="BGR", use_container_width=True)

    camera.release()