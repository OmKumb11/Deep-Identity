import os
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image, ImageChops, ImageEnhance
from mtcnn import MTCNN
import tempfile

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

st.set_page_config(
    page_title="Deep-Identity: Forensic Suite",
    layout="wide",
    page_icon="🛡️"
)

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
                st.caption(
                    f"Face detected — detector confidence: {faces[0]['confidence']:.2%}"
                )
    except Exception:
        pass

    if not face_detected:
        st.warning("No face detected — analyzing full frame")

    img_uint8 = cv2.resize(rgb, (299, 299))
    img_batch = preprocess(img_uint8)
    pred = float(model.predict(img_batch, verbose=0)[0][0])
    heatmap_color = get_gradcam(grad_model, img_batch)
    overlay = cv2.addWeighted(img_uint8, 0.6, heatmap_color, 0.4, 0)
    ela = run_ela(img_uint8)
    return pred, overlay, ela, face_detected

# ── UI ──
st.title("🛡️ Deep-Identity: Neural Forensic Suite")
st.caption(
    "Deepfake detection using Xception-Net + Grad-CAM + ELA | "
    "Trained on FaceForensics++ C23 | AUC: 0.9504"
)

with st.sidebar:
    st.header("Input")
    input_method = st.radio("Method", ("Upload File", "Manual Path"))
    if input_method == "Upload File":
        uploaded = st.file_uploader(
            "Choose file", type=["mp4", "avi", "jpg", "jpeg", "png"]
        )
    else:
        manual_path = st.text_input("Paste full file path")

    st.markdown("---")
    st.markdown("**Model:** Xception-Net fine-tuned on FF++ C23")
    st.markdown("**Overall AUC:** 0.9504")
    st.markdown("**Accuracy:** 88.91%")
    st.markdown("**Per-type AUC**")
    st.markdown("""
    | Type | AUC |
    |------|-----|
    | DeepFakeDetection | 1.0000 |
    | FaceShifter | 0.9999 |
    | FaceSwap | 0.9998 |
    | NeuralTextures | 0.9999 |
    | Face2Face | 0.9196 |
    | Deepfakes | 0.7767 |
    """)

with st.spinner("Loading forensic engine..."):
    model, grad_model, face_detector = load_model()

frame = None

if input_method == "Upload File" and uploaded:
    ext = os.path.splitext(uploaded.name)[1].lower()
    if ext in [".mp4", ".avi"]:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        cap = cv2.VideoCapture(tmp_path)
        ret, frame = cap.read()
        cap.release()
        os.unlink(tmp_path)
        if not ret:
            st.error("Could not read video frame.")
    else:
        arr = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

elif input_method == "Manual Path" and manual_path:
    ext = os.path.splitext(manual_path)[1].lower()
    if os.path.exists(manual_path):
        if ext in [".mp4", ".avi"]:
            cap = cv2.VideoCapture(manual_path)
            ret, frame = cap.read()
            cap.release()
        else:
            frame = cv2.imread(manual_path)
    else:
        st.error("File not found — check the path")

if frame is not None:
    with st.spinner("Analyzing..."):
        pred, overlay, ela, face_detected = analyze_frame(
            frame, model, grad_model, face_detector
        )

    is_fake = pred > 0.5
    confidence = pred if is_fake else 1 - pred

    if is_fake:
        st.error(f"🚨 MANIPULATION DETECTED — {confidence:.2%} confidence")
    else:
        st.success(f"✅ AUTHENTIC — {confidence:.2%} confidence")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original frame")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(cv2.resize(rgb, (299, 299)), use_container_width=True)

    with col2:
        st.subheader("Grad-CAM attention")
        st.image(overlay, use_container_width=True)
        st.caption("Red = regions driving the prediction")

    with col3:
        st.subheader("ELA map")
        st.image(ela, use_container_width=True)
        st.caption("Bright = compression inconsistencies")

    with st.expander("Raw scores"):
        st.json({
            "fake_probability": round(pred, 4),
            "real_probability": round(1 - pred, 4),
            "verdict": "FAKE" if is_fake else "REAL",
            "face_detected": face_detected
        })