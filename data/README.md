# Data Directory

This directory contains the competition dataset.

## Expected Structure

```
data/
├── train/
│   ├── samples/
│   │   ├── video_001/
│   │   │   ├── drone_video.mp4
│   │   │   └── object_images/
│   │   │       ├── img_1.jpg
│   │   │       ├── img_2.jpg
│   │   │       └── img_3.jpg
│   │   └── ...
│   └── annotations/
│       └── annotations.json
├── test/
│   └── samples/
│       └── ...
└── data.yaml  # YOLO training configuration
```

## Download Dataset

Download the Zalo AI Challenge 2025 dataset and extract it into this directory.

### Option 1: Google Drive folder

Download from:
https://drive.google.com/drive/folders/1fcDnRgNIE6XZw1ppbLczHo5e2q1n8y6h?usp=sharing

After downloading, ensure the final layout matches the tree above (i.e. `data/train/...` and `data/test/...`).

### Option 2: Competition website

Download from the competition website and extract here.

## YOLO Training Configuration

Create `data.yaml` for training:

```yaml
# data.yaml
path: /absolute/path/to/Zalo-AI-DLUS/data
train: train/images
val: val/images

nc: 1  # number of classes
names: ['target']
```

**Note**: This directory is gitignored to avoid committing large files.
