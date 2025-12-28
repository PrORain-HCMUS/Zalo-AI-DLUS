# Training Guide

## Overview

This guide explains how to train the YOLOv8 object detection model for the AeroEyes challenge.

## Prerequisites

- Dataset prepared in YOLO format
- GPU with at least 8GB VRAM
- Virtual environment activated

## Dataset Preparation

### 1. Convert to YOLO Format

Your dataset should be organized as:

```
data/
├── images/
│   ├── train/
│   │   ├── img_001.jpg
│   │   └── ...
│   └── val/
│       ├── img_100.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── img_001.txt
    │   └── ...
    └── val/
        ├── img_100.txt
        └── ...
```

### 2. Create data.yaml

```yaml
# data.yaml
path: /absolute/path/to/Zalo-AI-DLUS/data
train: images/train
val: images/val

nc: 1  # number of classes
names: ['target']
```

## Training Commands

### Basic Training

```bash
python src/train.py \
  --data data.yaml \
  --model yolov8s.pt \
  --epochs 100 \
  --img-size 640 \
  --batch-size 16
```

### Advanced Training

```bash
python src/train.py \
  --data data.yaml \
  --model yolov8s.pt \
  --epochs 200 \
  --img-size 640 \
  --batch-size 32 \
  --project runs/train \
  --name yolov8s_aeroeyes_v2
```

## Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data` | Required | Path to data.yaml |
| `--model` | yolov8s.pt | Pretrained model |
| `--epochs` | 100 | Number of epochs |
| `--img-size` | 640 | Input image size |
| `--batch-size` | 16 | Batch size |
| `--project` | runs/train | Project directory |
| `--name` | yolov8s_aeroeyes | Experiment name |

## Model Selection

| Model | Parameters | Speed | Accuracy |
|-------|------------|-------|----------|
| yolov8n | 3.2M | Fastest | Lower |
| yolov8s | 11.2M | Fast | Good |
| yolov8m | 25.9M | Medium | Better |
| yolov8l | 43.7M | Slow | Best |

**Recommendation**: Use YOLOv8s for balance between speed and accuracy.

## Monitoring Training

Training results are saved in `runs/train/yolov8s_aeroeyes/`:

```
runs/train/yolov8s_aeroeyes/
├── weights/
│   ├── best.pt      # Best model
│   └── last.pt      # Last epoch
├── results.png      # Training curves
├── confusion_matrix.png
└── args.yaml        # Training arguments
```

### TensorBoard

```bash
tensorboard --logdir runs/train
```

## After Training

### 1. Copy Best Weights

```bash
cp runs/train/yolov8s_aeroeyes/weights/best.pt checkpoints/best.pt
```

### 2. Validate Model

```bash
python -c "from ultralytics import YOLO; model = YOLO('checkpoints/best.pt'); model.val(data='data.yaml')"
```

### 3. Test Inference

```bash
python src/predict.py \
  --video test_video.mp4 \
  --ref-images ref1.jpg ref2.jpg ref3.jpg \
  --output test_predictions.json
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```bash
python src/train.py --data data.yaml --batch-size 8
```

### Slow Training

Use smaller model:
```bash
python src/train.py --data data.yaml --model yolov8n.pt
```

### Poor Accuracy

- Increase epochs: `--epochs 200`
- Use data augmentation (enabled by default)
- Collect more training data
- Use larger model: `--model yolov8m.pt`

## Tips

1. **Start small**: Train for 50 epochs first to verify setup
2. **Monitor metrics**: Watch mAP50 and mAP50-95
3. **Early stopping**: Training stops if no improvement for 50 epochs
4. **Save checkpoints**: Best model is automatically saved
5. **Use pretrained weights**: Always start from pretrained model
