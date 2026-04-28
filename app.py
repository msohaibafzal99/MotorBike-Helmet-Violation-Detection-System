import streamlit as st
from ultralytics import YOLO
import numpy as np
import tempfile
from PIL import Image
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

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

# ---------------- IMAGE ----------------
if input_type == "Image":

    st.markdown("""
    <h2 style='text-align:center; color:#38bdf8;'>📷 Image Detection</h2>
    <p style='text-align:center; color:#94a3b8;'>
    ⚠️ Upload a clear image with motorbike and rider for best accuracy
    </p>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        results = model(image_np)[0]
        annotated = results.plot()  # YOLO built-in drawing

        st.image(annotated, channels="RGB", use_container_width=True)

# ---------------- VIDEO ----------------
elif input_type == "Video":

    st.markdown("""
    <h2 style='text-align:center; color:#38bdf8;'>🎥 Video Detection</h2>
    <p style='text-align:center; color:#94a3b8;'>
    Upload a video containing motorbike scenes
    </p>
    """, unsafe_allow_html=True)

    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        import cv2  # only for video reading (safe use)

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)[0]
            frame = results.plot()

            stframe.image(frame, channels="RGB", use_container_width=True)

        cap.release()

# ---------------- WEBCAM (MOBILE + CLOUD SAFE) ----------------
elif input_type == "Webcam":

    st.markdown("""
    <h2 style='text-align:center; color:#38bdf8;'>📸 Live Mobile / Webcam Detection</h2>
    <p style='text-align:center; color:#94a3b8;'>
    Works on mobile phone & laptop browser camera
    </p>
    """, unsafe_allow_html=True)

    class VideoProcessor(VideoTransformerBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")

            results = model(img)[0]
            annotated = results.plot()

            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(
        key="helmet-detection",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )