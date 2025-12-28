# Checkpoints Directory

This directory contains trained model weights.

## Required Files

- `best.pt` - Trained YOLOv8s model weights

## Download Weights

### Option 1: From Google Drive
```bash
# 1) Open this folder in your browser and download the checkpoint file(s):
# https://drive.google.com/drive/folders/1fcDnRgNIE6XZw1ppbLczHo5e2q1n8y6h?usp=sharing
#
# 2) Place the downloaded YOLO checkpoint at:
# checkpoints/best.pt
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
