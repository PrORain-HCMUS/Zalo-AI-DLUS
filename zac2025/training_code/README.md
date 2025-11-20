# Training Code - AeroEyes

This folder contains the complete training pipeline and preprocessing code for the AeroEyes drone object detection system.

## Contents

- `train.py` - Training script for YOLOv8 model
- `preprocessing.ipynb` - Data preprocessing and augmentation notebook

## Training YOLOv8

### Requirements

Prepare your dataset in YOLO format:
```
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

### data.yaml Format

```yaml
path: /path/to/dataset
train: images/train
val: images/val

nc: 1
names: ['object']
```

### Training Command

```bash
python train.py --data dataset/data.yaml --epochs 100 --batch-size 16
```

### Arguments

- `--data` - Path to data.yaml configuration file
- `--model` - Base model (default: yolov8s.pt)
- `--epochs` - Number of training epochs (default: 100)
- `--img-size` - Input image size (default: 640)
- `--batch-size` - Batch size (default: 16)

## Model Export

After training, export the model:

```bash
yolo export model=runs/train/yolov8s_aeroeyes/weights/best.pt format=torchscript
```

For Jetson deployment:
```bash
yolo export model=best.pt format=engine half=True device=0
```

## Preprocessing

The preprocessing notebook contains utilities for:
- Data augmentation
- Dataset preparation
- Bounding box visualization
- Training/validation split

## Model Architecture

**YOLOv8s Specifications:**
- Parameters: ~11M
- Input size: 640x640
- Backbone: CSPDarknet
- Neck: PANet
- Head: Decoupled head

## Hyperparameters

Key training parameters:
- Learning rate: 0.01
- Optimizer: SGD
- Momentum: 0.937
- Weight decay: 0.0005
- Warmup epochs: 3

## Performance Optimization

For better results:
1. Use pre-trained weights on COCO
2. Apply mosaic augmentation
3. Use mixed precision training (fp16)
4. Fine-tune on domain-specific data

## Output

Trained model will be saved to:
```
runs/train/yolov8s_aeroeyes/weights/best.pt
```

Copy this to `saved_models/best.pt` in the project root for inference.
