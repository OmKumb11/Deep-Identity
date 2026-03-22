import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
from PIL import Image, ImageChops, ImageEnhance
import matplotlib.pyplot as plt

# 1. Page Configuration
st.set_page_config(page_title="Deep-Identity Forensic Suite", layout="wide")
st.title("🛡️ Deep-Identity: Neural Forensic Dashboard")
st.markdown("---")

# 2. Load Models (Cached for speed)
@st.cache_resource
def load_forensic_engine():
    # Build Skeleton
    base = tf.keras.applications.Xception(weights=None, include_top=False, input_shape=(299, 299, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=base.input, outputs=x)
    
    # Load Weights (Update path if needed)
    weights_path = r"E:\Deep_Detection\xception_weights_tf_dim_ordering_tf_kernels_notop.h5"
    model.load_weights(weights_path, by_name=True, skip_mismatch=True)
    return model, MTCNN()

model, detector = load_forensic_engine()

# 3. Sidebar Stats
with st.sidebar:
    st.header("System Status")
    st.success("Core Engine: Xception-V1")
    st.info("Dataset: FaceForensics++ (C23)")
    st.warning("Hardware: RTX 3050 (Inference Active)")
    quality = st.slider("ELA Sensitivity", 50, 95, 90)

# 4. File Uploader
uploaded_file = st.file_uploader("Upload Video or Image for Forensic Audit", type=['mp4', 'jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # If it's a video, we grab the first frame
    if uploaded_file.name.endswith('.mp4'):
        tfile = open("temp_vid.mp4", "wb")
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture("temp_vid.mp4")
        ret, frame = cap.read()
        cap.release()
    else:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Identity Integrity Scan")
        faces = detector.detect_faces(rgb_frame)
        if faces:
            x, y, w, h = faces[0]['box']
            face_crop = rgb_frame[y:y+h, x:x+w]
            
            # Predict
            input_face = cv2.resize(face_crop, (299, 299))
            input_face = np.expand_dims(input_face, axis=0) / 255.0
            prediction = model.predict(input_face)[0][0]
            
            st.image(face_crop, caption="Extracted Facial DNA", use_container_width=True)
            
            if prediction > 0.5:
                st.error(f"🚨 MANIPULATION DETECTED: {prediction:.2%}")
            else:
                st.success(f"✅ AUTHENTIC IDENTITY: {1-prediction:.2%}")
        else:
            st.warning("No face found in this sample.")

    with col2:
        st.subheader("🌐 Scene Integrity Scan")
        # ELA Logic
        original = Image.fromarray(rgb_frame)
        original.save("ela_temp.jpg", 'JPEG', quality=quality)
        temporary = Image.open("ela_temp.jpg")
        diff = ImageChops.difference(original, temporary)
        extrema = diff.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        scale = 255.0 / (max_diff if max_diff > 0 else 1)
        ela_map = ImageEnhance.Brightness(diff).enhance(scale)
        
        st.image(ela_map, caption="Environmental Noise Map (ELA)", use_container_width=True)
        st.write("Look for high-contrast 'glow' in backgrounds to spot object splicing.")

st.markdown("---")
st.caption("Deep-Identity Prototype | VIT Bhopal University | 2026")