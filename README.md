# Deep-Identity: Neural Forensic Suite

A dual-channel deepfake detection system using Xception-Net + ELA, trained on FaceForensics++ C23.

## Results

| Metric | Value |
|--------|-------|
| Overall AUC-ROC | 0.9504 |
| Equal Error Rate | 0.1158 |
| Accuracy | 88.91% |

### Per-manipulation AUC
| Type | AUC |
|------|-----|
| DeepFakeDetection | 1.0000 |
| FaceShifter | 0.9999 |
| FaceSwap | 0.9998 |
| NeuralTextures | 0.9999 |
| Face2Face | 0.9196 |
| Deepfakes | 0.7767 |

## Architecture
- Face detection: MTCNN
- Backbone: Xception-Net (pretrained ImageNet, fine-tuned on FF++)
- Explainability: Grad-CAM + Error Level Analysis (ELA)
- Dataset: FaceForensics++ C23 — 360,867 frames across 7 categories

## Setup
```bash
git clone https://github.com/yourusername/Deep-Identity
cd Deep-Identity
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
```bash
streamlit run app.py
```

## Project Structure
```t
Deep-Identity/
├── app.py                  # Streamlit demo
├── step1_extract_frames.py # Frame extraction
├── step2_build_manifest.py # Dataset manifest
├── step3_train.py          # Model training
├── step4_evaluate.py       # Evaluation
├── step5_visualize.py      # Grad-CAM + ELA
├── manifest.csv            # Dataset index
├── eval_results.csv        # Test set metrics
└── requirements.txt
```