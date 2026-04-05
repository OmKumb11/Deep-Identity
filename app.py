import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
from PIL import Image, ImageChops, ImageEnhance
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Deep-Identity: Forensic Suite", layout="wide")

# 1. CACHED ENGINE LOADING 
# This stops the "infinite loading" by keeping the model in RAM
@st.cache_resource
def load_forensic_engine():
    # Update this path if you move your weights file
    weights_path = "xception_weights_tf_dim_ordering_tf_kernels_notop.h5"
    
    # Build Xception Skeleton
    base_model = tf.keras.applications.Xception(
        weights=None, 
        input_shape=(299, 299, 3), 
        include_top=False, 
        pooling='avg'
    )
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(base_model.output)
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
    
    detector = MTCNN()
    return model, detector

# 2. SCENE INTEGRITY LOGIC (ELA)
def run_scene_integrity_scan(image, quality=90):
    temp_file = 'ela_temp.jpg'
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    image.save(temp_file, 'JPEG', quality=quality)
    temp_image = Image.open(temp_file)
    
    ela_image = ImageChops.difference(image, temp_image)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0: max_diff = 1
    scale = 255.0 / max_diff
    
    return ImageEnhance.Brightness(ela_image).enhance(scale)

# --- USER INTERFACE ---
st.title("🛡️ Deep-Identity: Neural Forensic Suite")
st.markdown("---")

# Sidebar Setup
st.sidebar.header("📁 Data Input")
input_method = st.sidebar.radio("Choose Input Method", ("Upload File", "Manual Path (Bypass Drive Lag)"))

# Initialize the "Brain"
model, detector = load_forensic_engine()

uploaded_file = None
manual_path = ""
frame = None

# Handle Inputs
if input_method == "Upload File":
    uploaded_file = st.sidebar.file_uploader("Choose a sample...", type=['mp4', 'avi', 'jpg', 'png'])
else:
    manual_path = st.sidebar.text_input("Paste Path here (e.g. E:\\archive\\...)", placeholder="E:\\...")

# --- FORENSIC PROCESSING ---
if uploaded_file or manual_path:
    with st.spinner("⏳ Analyzing Neural Artifacts..."):
        # Load the Frame
        if manual_path:
            cap = cv2.VideoCapture(manual_path)
            ret, frame = cap.read()
            cap.release()
        elif uploaded_file:
            if uploaded_file.name.lower().endswith(('.mp4', '.avi')):
                with open("temp_sample.mp4", "wb") as f:
                    f.write(uploaded_file.read())
                cap = cv2.VideoCapture("temp_sample.mp4")
                ret, frame = cap.read()
                cap.release()
            else:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, 1)

        if frame is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            col1, col2 = st.columns(2)

            # COLUMN 1: AI IDENTITY AUDIT
            with col1:
                st.subheader("🔍 Identity Integrity Scan")
                faces = detector.detect_faces(rgb_frame)
                
                if faces:
                    x, y, w, h = faces[0]['box']
                    x, y = max(0, x), max(0, y) # Bounds safety
                    face_crop = rgb_frame[y:y+h, x:x+w]
                    
                    # Preprocessing for Xception-Net (-1 to 1)
                    input_face = cv2.resize(face_crop, (299, 299))
                    input_face = (input_face.astype(np.float32) / 127.5) - 1.0
                    input_face = np.expand_dims(input_face, axis=0)
                    
                    prediction_raw = model.predict(input_face)
                    prediction = float(prediction_raw[0][0])
                    
                    st.image(face_crop, caption="Extracted Facial DNA", use_container_width=True)
                    
                    if prediction > 0.5:
                        st.error(f"🚨 MANIPULATION DETECTED: {prediction:.2%}")
                    else:
                        st.success(f"✅ AUTHENTIC IDENTITY: {1-prediction:.2%}")
                else:
                    st.warning("No biometric signatures found in frame.")

            # COLUMN 2: ELA SCENE SCAN
            with col2:
                st.subheader("🌐 Scene Integrity Scan")
                ela_map = run_scene_integrity_scan(rgb_frame)
                st.image(rgb_frame, caption="Original Capture", use_container_width=True)
                st.image(ela_map, caption="Environmental Noise Map (ELA)", use_container_width=True)
                st.info("Visualizes compression inconsistencies in the background.")
        else:
            st.error("Input Failure: Check your file path or format.")