# AeroEyes - Drone Object Detection & Tracking

**Team DLUS - Zalo AI Challenge 2025**

A hybrid detection and tracking system for finding and localizing target objects in drone videos using reference images. The system combines YOLOv8 for object detection, DINOv2 for feature matching, and ByteTrack for temporal tracking.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Project Structure](#project-structure)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Results](#results)
- [Troubleshooting](#troubleshooting)

---

## Architecture Overview

### Model Components

| Component | Model | Parameters | Purpose |
|-----------|-------|------------|---------|
| **Detection** | YOLOv8s | ~11M | Detect all objects in frame |
| **Feature Matching** | DINOv2-small | ~22M | Match detections with reference images |
| **Tracking** | ByteTrack | <1M | Temporal smoothing & tracking |
| **Total** | | **~33M** | Fits Jetson Xavier NX (50M limit) |

### Pipeline Flow

```
Reference Images (3x) → DINOv2 → [Cached Features]
                                        ↓
Frame → YOLOv8s → All Detections → DINOv2 → Match with Reference
                                                    ↓
                                            Best Match → ByteTrack
                                                    ↓
                                            Tracked BBox → Output
```

---

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (tested on RTX 3090, Jetson Xavier NX)
- **RAM**: Minimum 16GB
- **Storage**: 10GB free space

### Software
- **Python**: 3.10.12 (tested and recommended)
- **CUDA**: 11.8 or higher
- **Operating System**: Ubuntu 20.04/22.04, Windows 10/11

---

## Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/PrORain-HCMUS/Zalo-AI-DLUS.git
cd Zalo-AI-DLUS
```

### Step 2: Create Virtual Environment

**Linux/Mac:**
```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install Cython
pip install -r requirements.txt
```

**Note**: If you encounter issues with `lap` or `cython-bbox`, install them separately:
```bash
pip install lap==0.4.0
pip install cython-bbox==0.1.3
```

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from ultralytics import YOLO; print('YOLOv8 OK')"
python -c "from transformers import AutoModel; print('Transformers OK')"
```

---

## Dataset Preparation

### Step 1: Download Dataset

Download the Zalo AI Challenge 2025 dataset from the competition website.
Download the extracted dataset from [Drive](https://drive.google.com/drive/u/0/folders/1r4vHwAfRj6OuMdOWPFqwntLhYLDDmpJf)

### Step 2: Extract and Organize

Extract the dataset and place it in the following structure:

```
Zalo-AI-DLUS/
├── data/
│   ├── extracted/          # YOLO format training data
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   └── val/
│   │       ├── images/
│   │       └── labels/
│   └── zalo/               # Original Zalo competition dataset
│       ├── train/
│       │   ├── samples/
│       │   │   ├── Backpack_0/
│       │   │   │   ├── drone_video.mp4
│       │   │   │   └── object_images/
│       │   │   │       ├── img_1.jpg
│       │   │   │       ├── img_2.jpg
│       │   │   │       └── img_3.jpg
│       │   │   └── ...
│       │   └── annotations/
│       │       └── annotations.json
│       └── test/
│           └── samples/
│               ├── BlackBox_0/
│               ├── BlackBox_1/
│               ├── CardboardBox_0/
│               ├── CardboardBox_1/
│               ├── LifeJacket_0/
│               ├── LifeJacket_1/
│               └── ...
```

**Commands:**
```bash
# Create data directory
mkdir -p data

# Extract dataset (example)
unzip zalo_ai_2025_dataset.zip -d data/

# Verify structure
ls -R data/train/samples/ | head -20
```

### Step 3: Download Pretrained Weights

Download the trained YOLOv8m model:

**Option 1: From Google Drive** (recommended)
```bash
# Install gdown
pip install gdown

# Download weights
gdown https://drive.google.com/file/d/1S14JGulatl9ysYefRnM8MmYCf82ZOgBg -O checkpoints/best.pt
```

Or download the weight from [link](https://drive.google.com/file/d/1S14JGulatl9ysYefRnM8MmYCf82ZOgBg)

**Option 2: Train from scratch** (see [Training](#training) section)

Place the weights file at:
```
checkpoints/best.pt
```

---

## Project Structure

```
Zalo-AI-DLUS/
├── src/                          # Source code
│   ├── models/                   # Model definitions
│   │   ├── __init__.py
│   │   ├── detector.py           # YOLOv8 detector
│   │   ├── feature_extractor.py  # DINOv2 feature extractor
│   │   └── tracker.py            # ByteTrack & SimpleTracker
│   ├── utils/                    # Utility functions
│   │   ├── __init__.py
│   │   └── inference_utils.py    # Helper functions
│   ├── predict.py                # Single video inference
│   ├── batch_predict.py          # Batch processing
│   └── train.py                  # Training script
├── config/                       # Configuration files
│   └── config.yaml               # Main configuration
├── checkpoints/                  # Model weights (gitignored)
│   └── best.pt                   # Trained YOLOv8 model
├── data/                         # Dataset (gitignored)
├── results/                      # Output results (gitignored)
├── docs/                         # Documentation
├── scripts/                      # Helper scripts
├── notebooks/                    # Jupyter notebooks
├── requirements.txt              # Python dependencies
├── source.txt                    # GitHub repository URL
└── README.md                     # This file
```

---

## Training

### Step 1: Prepare Training Data

Create a YOLO format dataset configuration file `data.yaml` in the root directory:

```yaml
# data.yaml
path: /data
train: train/images
val: val/images

nc: 1  # number of classes
names: ['target']
```

### Step 2: Train YOLOv8

```bash
python src/train.py \
  --data data.yaml \
  --model yolov8m.pt \
  --epochs 100 \
  --img-size 640 \
  --batch-size 16 \
  --project runs/train \
  --name yolov8s_aeroeyes
```

**Training Parameters:**
- `--data`: Path to data.yaml configuration
- `--model`: Pretrained model (yolov8n.pt, yolov8s.pt, yolov8m.pt)
- `--epochs`: Number of training epochs (default: 100)
- `--img-size`: Input image size (default: 640)
- `--batch-size`: Batch size (adjust based on GPU memory)
- `--project`: Project directory for saving results
- `--name`: Experiment name

### Step 3: Copy Best Weights

After training, copy the best weights to the checkpoints directory:

```bash
cp runs/train/yolov8m_aeroeyes/weights/best.pt checkpoints/best.pt
```

---

## Inference

### Single Video Inference

Process a single drone video with reference images:

```bash
python src/predict.py \
  --video data/zalo/test/samples/BlackBox_0/drone_video.mp4 \
  --ref-images data/zalo/test/samples/BlackBox_0/object_images/img_1.jpg \
              data/zalo/test/samples/BlackBox_0/object_images/img_2.jpg \
              data/zalo/test/samples/BlackBox_0/object_images/img_3.jpg \
  --config config/config.yaml \
  --output results/video_BlackBox_0_predictions.json \
  --visualize
```

**Arguments:**
- `--video`: Path to drone video file
- `--ref-images`: Paths to 3 reference images (space-separated)
- `--config`: Configuration file (default: config/config.yaml)
- `--output`: Output JSON file path
- `--visualize`: Generate visualization video (optional)

**Output Format:**
```json
[
  {"frame": 370, "x1": 422, "y1": 310, "x2": 470, "y2": 355},
  {"frame": 371, "x1": 424, "y1": 312, "x2": 468, "y2": 354}
]
```

### Batch Processing (Competition Submission)

Process entire competition dataset:

```bash
python src/batch_predict.py \
  --dataset data/zalo/test \
  --output submission.json \
  --config config/config.yaml \
  --visualize
```

**Arguments:**
- `--dataset`: Path to dataset root (contains samples/ directory)
- `--output`: Output submission JSON file
- `--config`: Configuration file
- `--visualize`: Generate visualization videos (optional)

**Output Format:**
```json
[
  {
    "video_id": "video_001",
    "detections": [
      {
        "bboxes": [
          {"frame": 370, "x1": 422, "y1": 310, "x2": 470, "y2": 355}
        ]
      }
    ]
  }
]
```

---

## Evaluation

### Calculate Metrics

To evaluate predictions against ground truth:
Đang so sánh với dummy result
```bash
python src/evaluate.py \
  --predictions submission.json \
  --ground-truth data/zalo/train/annotations/annotations.json \
  --output results/evaluation_results.json \
  --iou-threshold 0.5
```


**Metrics:**
- **ST-IoU**: Spatio-Temporal Intersection over Union
- **Precision**: Detection precision
- **Recall**: Detection recall
- **F1-Score**: Harmonic mean of precision and recall

---

## Configuration

Edit `config/config.yaml` to adjust model parameters:

```yaml
models:
  yolo:
    type: "yolov8"
    weights: "checkpoints/best.pt"
    img_size: 640
    conf_threshold: 0.20      # Detection confidence threshold
    iou_threshold: 0.45       # NMS IoU threshold

  dinov2:
    model_name: "facebook/dinov2-small"

inference:
  matching_confidence_threshold: 0.60  # DINOv2 similarity threshold
  tracking_confidence_threshold: 0.40  # Min confidence for tracking
  redetection_lost_frames: 20          # Frames before switching to search
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `conf_threshold` | 0.20 | YOLO confidence threshold |
| `iou_threshold` | 0.45 | NMS IoU threshold |
| `matching_confidence_threshold` | 0.60 | DINOv2 similarity threshold |
| `tracking_confidence_threshold` | 0.40 | Min confidence to stay in tracking |
| `redetection_lost_frames` | 20 | Frames to wait before re-searching |

---

## Results

### Competition Performance

| Metric | Value |
|--------|-------|
| **ST-IoU** | 0.XXX |
| **Precision** | 0.XXX |
| **Recall** | 0.XXX |
| **F1-Score** | 0.XXX |

### Performance on Jetson Xavier NX

| Mode | FPS | Description |
|------|-----|-------------|
| **Tracking (High Conf)** | 35-40 | Detection every 15 frames |
| **Tracking (Medium Conf)** | 30-35 | Detection every 10 frames |
| **Search Mode** | 20-25 | Detection every frame |

---

## Troubleshooting

### Issue: CUDA out of memory

**Solution:** Reduce batch size or image size in config:
```yaml
yolo:
  img_size: 480  # Instead of 640
```

### Issue: Low detection accuracy

**Solutions:**
1. Lower `matching_confidence_threshold` (e.g., 0.50)
2. Increase `redetection_lost_frames` (e.g., 30)
3. Fine-tune YOLO on your specific dataset

### Issue: Slow inference

**Solutions:**
1. Convert models to TensorRT:
```bash
yolo export model=checkpoints/best.pt format=engine half=True device=0
```
2. Increase detection intervals in config
3. Use smaller model (yolov8n instead of yolov8s)

### Issue: Import errors

**Solution:** Ensure you're in the virtual environment and run from project root:
```bash
source .venv/bin/activate  # Linux/Mac
cd /path/to/Zalo-AI-DLUS
python src/predict.py ...
```

### Issue: Dataset path errors (cached directory)

**Problem:** YOLOv8 shows errors like:
```
RuntimeError: Dataset 'data.yaml' error
Dataset 'data.yaml' images not found, missing path 'D:\old\path\datasets\data\val\images'
```

**Cause:** YOLOv8 cached the old dataset directory path in its settings, even after you moved/renamed directories.

**Solutions:**
1. **Update YOLOv8 settings** (recommended):
```bash
python -c "from ultralytics import settings; settings.update({'datasets_dir': 'D:\\your\\project\\path'}); print('Settings updated')"
```

---

## Citation

If you use this code, please cite:

```bibtex
@misc{aeroeyes2025,
  title={AeroEyes: Hybrid Detection and Tracking for Drone Videos},
  author={Team DLUS},
  year={2025},
  howpublished={Zalo AI Challenge 2025}
}
```

---

## License

MIT License

---

## Contact

For questions or issues:
- **GitHub Issues**: https://github.com/PrORain-HCMUS/Zalo-AI-DLUS/issues
- **Email**: llehoangvum10@gmail.com

---

## Acknowledgments

- YOLOv8: [Ultralytics](https://github.com/ultralytics/ultralytics)
- DINOv2: [Meta AI](https://github.com/facebookresearch/dinov2)
- ByteTrack: [ByteTrack](https://github.com/ifzhang/ByteTrack)
- Zalo AI Challenge 2025
