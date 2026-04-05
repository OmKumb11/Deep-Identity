import os
import csv
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image, ImageChops, ImageEnhance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

MODEL_PATH  = "best_model.h5"
MANIFEST    = "manifest.csv"
IMG_SIZE    = 299
OUTPUT_DIR  = "explainability_outputs"
SAMPLES_PER_TYPE = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# ── Grad-CAM ──
grad_model = tf.keras.Model(
    inputs=model.input,
    outputs=[model.get_layer("block14_sepconv2_bn").output, model.output]
)

def get_gradcam(img_array):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val == 0:
        heatmap = tf.ones_like(heatmap) * 0.5
    else:
        heatmap = heatmap / max_val
    heatmap = heatmap.numpy()
    if heatmap.ndim == 0:
        heatmap = np.ones((10, 10)) * float(heatmap)
    return heatmap
# ── ELA ──
def run_ela(img_array_uint8, quality=90):
    pil_img = Image.fromarray(img_array_uint8).convert("RGB")
    temp_path = "ela_temp_vis.jpg"
    pil_img.save(temp_path, "JPEG", quality=quality)
    reloaded = Image.open(temp_path).convert("RGB")
    ela = ImageChops.difference(pil_img, reloaded)
    max_diff = max(ex[1] for ex in ela.getextrema()) or 1
    ela = ImageEnhance.Brightness(ela).enhance(255.0 / max_diff)
    os.remove(temp_path)
    return np.array(ela)

# ── load samples ──
def load_samples():
    samples = {}
    with open(MANIFEST) as f:
        for row in csv.DictReader(f):
            if row["split"] == "test":
                t = row["type"]
                if t not in samples:
                    samples[t] = []
                if len(samples[t]) < SAMPLES_PER_TYPE:
                    samples[t].append({
                        "path":  row["path"],
                        "label": int(row["label"]),
                        "type":  t
                    })
    return samples

samples = load_samples()
print(f"Generating visualizations for {len(samples)} types...")

for type_name, type_samples in samples.items():
    for i, sample in enumerate(type_samples):
        try:
            # load image
            raw = tf.io.read_file(sample["path"])
            img = tf.image.decode_jpeg(raw, channels=3)
            img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
            img_uint8 = tf.cast(img, tf.uint8).numpy()
            img_norm  = (tf.cast(img, tf.float32) / 127.5 - 1.0)
            img_batch = tf.expand_dims(img_norm, 0)

            # predict
            pred = float(model.predict(img_batch, verbose=0)[0][0])
            label_str = "FAKE" if sample["label"] == 1 else "REAL"
            pred_str  = f"{'FAKE' if pred > 0.5 else 'REAL'} ({pred:.2%})"

            # grad-cam
            heatmap = get_gradcam(img_batch)
            heatmap_resized = cv2.resize(heatmap.astype(np.float32), (IMG_SIZE, IMG_SIZE))
            heatmap_color = cv2.applyColorMap(
                np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
            )
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(img_uint8, 0.6, heatmap_color, 0.4, 0)

            # ela
            ela_img = run_ela(img_uint8)

            # plot
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            fig.suptitle(
                f"{type_name} | Ground truth: {label_str} | Prediction: {pred_str}",
                fontsize=12
            )

            axes[0].imshow(img_uint8)
            axes[0].set_title("Original frame")
            axes[0].axis("off")

            axes[1].imshow(overlay)
            axes[1].set_title("Grad-CAM attention")
            axes[1].axis("off")

            axes[2].imshow(ela_img)
            axes[2].set_title("ELA map")
            axes[2].axis("off")

            plt.tight_layout()
            out_path = os.path.join(
                OUTPUT_DIR, f"{type_name}_sample{i+1}.png"
            )
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved: {out_path}")

        except Exception as e:
            print(f"Failed {type_name} sample {i+1}: {e}")

print(f"\nDone. All visualizations saved to {OUTPUT_DIR}/")
print("These are your explainability figures for the paper.")