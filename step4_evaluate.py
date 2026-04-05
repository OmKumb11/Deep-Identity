import os
import csv
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

MODEL_PATH = "best_model.h5"
MANIFEST   = "manifest.csv"
IMG_SIZE   = 299
BATCH_SIZE = 2

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

def load_test():
    paths, labels, types = [], [], []
    with open(MANIFEST) as f:
        for row in csv.DictReader(f):
            if row["split"] == "test":
                paths.append(row["path"])
                labels.append(int(row["label"]))
                types.append(row["type"])
    return paths, labels, types

print("Loading test set...")
paths, labels, types = load_test()
print(f"Test samples: {len(paths)}")

def predict_batch(batch_paths):
    imgs = []
    for p in batch_paths:
        try:
            img = tf.image.decode_jpeg(tf.io.read_file(p), channels=3)
            img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
            img = tf.cast(img, tf.float32) / 127.5 - 1.0
            imgs.append(img)
        except Exception:
            imgs.append(tf.zeros([IMG_SIZE, IMG_SIZE, 3]))
    return model.predict(tf.stack(imgs), verbose=0).flatten()

print("Running inference...")
preds = []
for i in range(0, len(paths), BATCH_SIZE):
    batch = paths[i:i+BATCH_SIZE]
    preds.extend(predict_batch(batch))
    if i % 500 == 0:
        print(f"  {i}/{len(paths)}")
preds = np.array(preds)
labels = np.array(labels)

# ── Overall AUC ──
overall_auc = roc_auc_score(labels, preds)
print(f"\n{'='*40}")
print(f"Overall AUC-ROC:  {overall_auc:.4f}")

# ── EER ──
fpr, tpr, thresholds = roc_curve(labels, preds)
fnr = 1 - tpr
eer_idx = np.argmin(np.abs(fpr - fnr))
eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
print(f"Equal Error Rate: {eer:.4f}  (lower = better)")

# ── Accuracy at 0.5 threshold ──
preds_binary = (preds > 0.5).astype(int)
accuracy = np.mean(preds_binary == labels)
print(f"Accuracy:         {accuracy:.4f}")

# ── Confusion matrix ──
cm = confusion_matrix(labels, preds_binary)
print(f"\nConfusion Matrix:")
print(f"  TN: {cm[0][0]}  FP: {cm[0][1]}")
print(f"  FN: {cm[1][0]}  TP: {cm[1][1]}")

# ── AUC per manipulation type ──
print(f"\nAUC per manipulation type:")
unique_types = sorted(set(types))
results = [["metric", "value"]]
results.append(["overall_auc", f"{overall_auc:.4f}"])
results.append(["eer", f"{eer:.4f}"])
results.append(["accuracy", f"{accuracy:.4f}"])

for t in unique_types:
    if t == "real":
        continue
    idx = [i for i, x in enumerate(types) if x == t or x == "real"]
    sub_labels = labels[idx]
    sub_preds  = preds[idx]
    if len(set(sub_labels)) < 2:
        continue
    auc = roc_auc_score(sub_labels, sub_preds)
    print(f"  {t:22s}: {auc:.4f}")
    results.append([f"auc_{t}", f"{auc:.4f}"])

# ── Save ──
with open("eval_results.csv", "w", newline="") as f:
    csv.writer(f).writerows(results)

print(f"\nSaved eval_results.csv")
print(f"{'='*40}")
print(f"Your paper headline number: AUC = {overall_auc:.4f}")