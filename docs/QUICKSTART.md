# Quick Start Guide

## 1. Setup (5 minutes)

```bash
# Clone and enter directory
git clone https://github.com/PrORain-HCMUS/Zalo-AI-DLUS.git
cd Zalo-AI-DLUS

# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## 2. Download Weights

Place your trained YOLOv8 weights at:
```
checkpoints/best.pt
```

## 3. Run Inference

### Single Video
```bash
python src/predict.py \
  --video path/to/video.mp4 \
  --ref-images ref1.jpg ref2.jpg ref3.jpg \
  --output predictions.json
```

### Batch Processing
```bash
python src/batch_predict.py \
  --dataset data/test \
  --output submission.json
```

## 4. Training (Optional)

```bash
python src/train.py \
  --data data.yaml \
  --epochs 100 \
  --batch-size 16
```

That's it! 🚀
