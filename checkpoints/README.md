# Checkpoints Directory

This directory contains trained model weights.

## Required Files

- `best.pt` - Trained YOLOv8s model weights

## Download Weights

### Option 1: From Google Drive
```bash
pip install gdown
gdown https://drive.google.com/uc?id=YOUR_FILE_ID -O best.pt
```

### Option 2: From Competition Website
Download from the competition website and place here.

### Option 3: Train from Scratch
```bash
python src/train.py --data data.yaml --epochs 100
cp runs/train/yolov8s_aeroeyes/weights/best.pt checkpoints/best.pt
```

## File Structure

```
checkpoints/
├── best.pt              # YOLOv8s trained weights (required)
├── yolov8n.pt          # Optional: smaller model
└── README.md           # This file
```

**Note**: Model weights are gitignored and must be downloaded separately.
