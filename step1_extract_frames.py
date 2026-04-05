import os
import cv2
from pathlib import Path

FF_ROOT    = r"E:\archive\FaceForensics++_C23"
OUTPUT_DIR = r"E:\Extracted_Frames"
FRAME_INTERVAL = 10
IMG_SIZE   = 299

FAKE_TYPES = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "DeepFakeDetection", "FaceShifter"]
REAL_TYPE  = "original"

def extract_from_video(video_path, out_dir):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Could not open: {video_path}")
        return 0

    os.makedirs(out_dir, exist_ok=True)
    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % FRAME_INTERVAL == 0:
            try:
                h, w = frame.shape[:2]
                # center crop to square then resize
                size = min(h, w)
                top  = (h - size) // 2
                left = (w - size) // 2
                crop = frame[top:top+size, left:left+size]
                crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
                out_path = os.path.join(out_dir, f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(out_path, crop)
                saved += 1
            except Exception:
                pass
        frame_idx += 1

    cap.release()
    return saved

total = 0

# ── Real videos ──
real_root = Path(FF_ROOT) / REAL_TYPE
for video in real_root.rglob("*.mp4"):
    rel = video.stem
    out_dir = Path(OUTPUT_DIR) / "real" / rel
    if out_dir.exists() and len(list(out_dir.glob("*.jpg"))) > 0:
        print(f"skip | real | {rel}")
        continue
    n = extract_from_video(video, str(out_dir))
    print(f"real | {rel} | {n} frames")
    total += n

# ── Fake videos ──
for fake_type in FAKE_TYPES:
    fake_root = Path(FF_ROOT) / fake_type
    if not fake_root.exists():
        print(f"Skipping {fake_type} — folder not found")
        continue
    for video in fake_root.rglob("*.mp4"):
        rel = video.stem
        out_dir = Path(OUTPUT_DIR) / fake_type / rel
        if out_dir.exists() and len(list(out_dir.glob("*.jpg"))) > 0:
            print(f"skip | {fake_type} | {rel}")
            continue
        n = extract_from_video(video, str(out_dir))
        print(f"{fake_type} | {rel} | {n} frames")
        total += n

print(f"\nDone. Total faces saved: {total}")
print(f"Output at: {OUTPUT_DIR}")