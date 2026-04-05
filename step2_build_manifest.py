import os, csv, random
from pathlib import Path

FRAMES_DIR = r"E:\Extracted_Frames"   # same as OUTPUT_DIR above
OUTPUT_CSV = "manifest.csv"
SEED = 42
random.seed(SEED)

FAKE_TYPES = {"Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"}
IMG_EXTS   = {".jpg", ".jpeg", ".png"}

all_rows = []

for type_dir in Path(FRAMES_DIR).iterdir():
    if not type_dir.is_dir():
        continue
    type_name = type_dir.name
    label = 0 if type_name == "real" else 1

    for img_path in type_dir.rglob("*"):
        if img_path.suffix.lower() in IMG_EXTS:
            all_rows.append({
                "path":  str(img_path),
                "label": label,
                "type":  type_name,
            })

random.shuffle(all_rows)
n = len(all_rows)
for i, row in enumerate(all_rows):
    if i < n * 0.70:   row["split"] = "train"
    elif i < n * 0.85: row["split"] = "val"
    else:               row["split"] = "test"

with open(OUTPUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["path","label","type","split"])
    w.writeheader()
    w.writerows(all_rows)

print(f"Manifest written: {OUTPUT_CSV}")
print(f"Total samples: {n}")

counts = {}
for r in all_rows:
    k = (r["split"], r["type"])
    counts[k] = counts.get(k, 0) + 1
for k, v in sorted(counts.items()):
    print(f"  {k[0]:6s} | {k[1]:20s} | {v} samples")