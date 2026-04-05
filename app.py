import os
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image, ImageChops, ImageEnhance
from mtcnn import MTCNN
import tempfile
def download_model():
    if not os.path.exists("best_model.h5"):
        import urllib.request
        url = "https://huggingface.co/Hades111106/deep-identity/resolve/main/best_model.h5"
        with st.spinner("Downloading model from HuggingFace..."):
            urllib.request.urlretrieve(url, "best_model.h5")

download_model()

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

st.set_page_config(
    page_title="Deep-Identity",
    layout="wide",
    page_icon="🛡️"
)

st.markdown("""
<style>
/* ── global ── */
[data-testid="stAppViewContainer"] {
    background: #0d0f1a;
    color: #e2e8f0;
}
[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] { display: none; }

/* ── top navbar ── */
.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.2rem 2rem;
    background: rgba(255,255,255,0.03);
    border-bottom: 1px solid rgba(139,92,246,0.2);
    margin-bottom: 2rem;
}
.navbar-brand {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    font-size: 1.3rem;
    font-weight: 700;
    color: #a78bfa;
    letter-spacing: 0.5px;
}
.navbar-badge {
    font-size: 0.7rem;
    background: rgba(139,92,246,0.15);
    border: 1px solid rgba(139,92,246,0.3);
    color: #a78bfa;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    font-weight: 500;
}
.navbar-stats {
    display: flex;
    gap: 2rem;
    font-size: 0.8rem;
    color: #94a3b8;
}
.navbar-stat span {
    color: #a78bfa;
    font-weight: 600;
}

/* ── hero ── */
.hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
}
.hero h1 {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.hero p {
    color: #64748b;
    font-size: 1rem;
    margin-bottom: 2rem;
}

/* ── upload card ── */
.upload-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(139,92,246,0.2);
    border-radius: 16px;
    padding: 2rem;
    margin: 0 auto 2rem;
    max-width: 700px;
}
.upload-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #a78bfa;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.5rem;
}

/* ── result banner ── */
.result-fake {
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    color: #fca5a5;
    font-size: 1.1rem;
    font-weight: 600;
    text-align: center;
    margin-bottom: 1.5rem;
}
.result-real {
    background: rgba(34,197,94,0.1);
    border: 1px solid rgba(34,197,94,0.3);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    color: #86efac;
    font-size: 1.1rem;
    font-weight: 600;
    text-align: center;
    margin-bottom: 1.5rem;
}

/* ── analysis cards ── */
.analysis-label {
    font-size: 0.7rem;
    font-weight: 600;
    color: #a78bfa;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.4rem;
}
.analysis-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(139,92,246,0.15);
    border-radius: 12px;
    padding: 1rem;
}
.analysis-caption {
    font-size: 0.72rem;
    color: #475569;
    margin-top: 0.4rem;
    text-align: center;
}

/* ── metrics row ── */
.metric-row {
    display: flex;
    gap: 1rem;
    justify-content: center;
    margin-bottom: 2rem;
    flex-wrap: wrap;
}
.metric-pill {
    background: rgba(139,92,246,0.08);
    border: 1px solid rgba(139,92,246,0.2);
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    text-align: center;
}
.metric-pill .val {
    font-size: 1.2rem;
    font-weight: 700;
    color: #a78bfa;
}
.metric-pill .lbl {
    font-size: 0.7rem;
    color: #64748b;
    margin-top: 0.1rem;
}

/* ── streamlit overrides ── */
[data-testid="stFileUploader"] {
    background: rgba(139,92,246,0.05) !important;
    border: 1px dashed rgba(139,92,246,0.3) !important;
    border-radius: 12px !important;
}
.stRadio label { color: #94a3b8 !important; font-size: 0.85rem !important; }
.stTextInput input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(139,92,246,0.3) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}
div[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(139,92,246,0.15) !important;
    border-radius: 10px !important;
}
</style>

<!-- navbar -->
<div class="navbar">
    <div class="navbar-brand">
        🛡️ Deep-Identity
        <span class="navbar-badge">v1.0</span>
    </div>
    <div class="navbar-stats">
        <div class="navbar-stat">AUC <span>0.9504</span></div>
        <div class="navbar-stat">Accuracy <span>88.91%</span></div>
        <div class="navbar-stat">EER <span>0.1158</span></div>
        <div class="navbar-stat">Dataset <span>FF++ C23</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── hero ──
st.markdown("""
<div class="hero">
    <h1>Neural Forensic Suite</h1>
    <p>Deepfake detection powered by Xception-Net · Grad-CAM explainability · Error Level Analysis</p>
</div>
""", unsafe_allow_html=True)

# ── metrics ──
st.markdown("""
<div class="metric-row">
    <div class="metric-pill"><div class="val">1.0000</div><div class="lbl">DeepFakeDetection</div></div>
    <div class="metric-pill"><div class="val">0.9999</div><div class="lbl">FaceShifter</div></div>
    <div class="metric-pill"><div class="val">0.9998</div><div class="lbl">FaceSwap</div></div>
    <div class="metric-pill"><div class="val">0.9999</div><div class="lbl">NeuralTextures</div></div>
    <div class="metric-pill"><div class="val">0.9196</div><div class="lbl">Face2Face</div></div>
    <div class="metric-pill"><div class="val">0.7767</div><div class="lbl">Deepfakes</div></div>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    model = tf.keras.models.load_model("best_model.h5")
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer("block14_sepconv2_bn").output, model.output]
    )
    detector = MTCNN()
    return model, grad_model, detector

def get_gradcam(grad_model, img_batch, img_size=299):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_batch)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(conv_outputs[0] * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val == 0:
        heatmap = tf.ones((10, 10), dtype=tf.float32) * 0.5
    else:
        heatmap = heatmap / max_val
    heatmap = heatmap.numpy().astype(np.float32)
    if heatmap.ndim == 0:
        heatmap = np.ones((10, 10), dtype=np.float32) * float(heatmap)
    heatmap = cv2.resize(heatmap, (img_size, img_size))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    return heatmap_color

def run_ela(img_uint8, quality=90):
    pil_img = Image.fromarray(img_uint8).convert("RGB")
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        pil_img.save(tmp_path, "JPEG", quality=quality)
        reloaded = Image.open(tmp_path).convert("RGB")
        ela = ImageChops.difference(pil_img, reloaded)
        max_diff = max(ex[1] for ex in ela.getextrema()) or 1
        ela = ImageEnhance.Brightness(ela).enhance(255.0 / max_diff)
        return np.array(ela)
    finally:
        os.unlink(tmp_path)

def preprocess(img_uint8, img_size=299):
    img = cv2.resize(img_uint8, (img_size, img_size))
    img = img.astype(np.float32) / 127.5 - 1.0
    return tf.expand_dims(img, 0)

def analyze_frame(frame_bgr, model, grad_model, face_detector):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    face_detected = False
    try:
        faces = face_detector.detect_faces(rgb)
        if faces:
            x, y, w, h = [max(0, i) for i in faces[0]['box']]
            if w > 20 and h > 20:
                rgb = rgb[y:y+h, x:x+w]
                face_detected = True
    except Exception:
        pass
    img_uint8 = cv2.resize(rgb, (299, 299))
    img_batch = preprocess(img_uint8)
    pred = float(model.predict(img_batch, verbose=0)[0][0])
    heatmap_color = get_gradcam(grad_model, img_batch)
    overlay = cv2.addWeighted(img_uint8, 0.6, heatmap_color, 0.4, 0)
    ela = run_ela(img_uint8)
    return pred, overlay, ela, face_detected

# ── input ──
with st.spinner("Loading forensic engine..."):
    model, grad_model, face_detector = load_model()

st.markdown('<div class="upload-card">', unsafe_allow_html=True)
st.markdown('<div class="upload-label">Input method</div>', unsafe_allow_html=True)
input_method = st.radio("", ("Upload File", "Manual Path"), horizontal=True, label_visibility="collapsed")

frame = None

if input_method == "Upload File":
    uploaded = st.file_uploader(
        "Drop an image or video file",
        type=["mp4", "avi", "jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    if uploaded:
        ext = os.path.splitext(uploaded.name)[1].lower()
        if ext in [".mp4", ".avi"]:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            cap = cv2.VideoCapture(tmp_path)
            ret, frame = cap.read()
            cap.release()
            os.unlink(tmp_path)
        else:
            arr = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
else:
    manual_path = st.text_input("Paste full file path", placeholder=r"E:\videos\sample.mp4")
    if manual_path and os.path.exists(manual_path):
        ext = os.path.splitext(manual_path)[1].lower()
        if ext in [".mp4", ".avi"]:
            cap = cv2.VideoCapture(manual_path)
            ret, frame = cap.read()
            cap.release()
        else:
            frame = cv2.imread(manual_path)
    elif manual_path:
        st.error("File not found — check the path")

st.markdown('</div>', unsafe_allow_html=True)

# ── results ──
if frame is not None:
    with st.spinner("Analyzing..."):
        pred, overlay, ela, face_detected = analyze_frame(
            frame, model, grad_model, face_detector
        )

    is_fake = pred > 0.5
    confidence = pred if is_fake else 1 - pred

    if not face_detected:
        st.warning("No face detected — analyzing full frame")

    if is_fake:
        st.markdown(
            f'<div class="result-fake">🚨 MANIPULATION DETECTED — {confidence:.2%} confidence</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="result-real">✅ AUTHENTIC — {confidence:.2%} confidence</div>',
            unsafe_allow_html=True
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="analysis-label">Original frame</div>', unsafe_allow_html=True)
        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(cv2.resize(rgb, (299, 299)), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="analysis-label">Grad-CAM attention</div>', unsafe_allow_html=True)
        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
        st.image(overlay, use_container_width=True)
        st.markdown('<div class="analysis-caption">Red = regions driving the prediction</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="analysis-label">ELA map</div>', unsafe_allow_html=True)
        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
        st.image(ela, use_container_width=True)
        st.markdown('<div class="analysis-caption">Bright = compression inconsistencies</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("Raw scores"):
        st.json({
            "fake_probability": round(pred, 4),
            "real_probability": round(1 - pred, 4),
            "verdict": "FAKE" if is_fake else "REAL",
            "face_detected": face_detected
        })