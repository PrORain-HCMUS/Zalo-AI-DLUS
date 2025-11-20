# AeroEyes - Drone Object Detection & Tracking

**Architecture: YOLOv8s + DINOv2 + ByteTrack**

Hybrid detection and tracking system for finding and localizing target objects in drone videos using reference images.

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

### Operating Modes

**1. Search Mode (Object Not Found)**
- Run YOLOv8s detection every frame
- Extract DINOv2 features for each detection
- Match against reference features
- Switch to tracking mode when found (similarity > 0.60)

**2. Tracking Mode (Object Being Tracked)**
- Run detection adaptively based on confidence:
  - High confidence (>0.80): every 15 frames
  - Medium confidence (>0.60): every 10 frames
  - Low confidence (>0.40): every 5 frames
- ByteTrack predicts bbox for non-detection frames
- Switch to search mode if lost for 20+ frames

---

## Project Structure

```
zac2025/
├── config/
│   └── config.yaml              # Configuration file
├── models/
│   ├── model_loader.py          # YOLOv8 + DINOv2 loaders
│   └── tracker.py               # ByteTrack + SimpleTracker
├── utils/
│   └── inference_utils.py       # Helper functions
├── saved_models/
│   └── best.pt                  # Your trained YOLOv8s model
├── predict.py                   # Main inference script
├── batch_predict.py             # Batch processing for competition
├── predict.sh                   # Convenience script
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

---

## Installation

### 1. Activate Virtual Environment

```bash
cd /home/lenovo/source/zac2025
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Models

Ensure your YOLOv8s model is at:
```
saved_models/best.pt
```

---

## Usage

### Single Video Inference

```bash
python predict.py \
  --video path/to/drone_video.mp4 \
  --ref-images ref1.jpg ref2.jpg ref3.jpg \
  --output predictions.json \
  --visualize
```

**Arguments:**
- `--video`: Path to drone video file
- `--ref-images`: Paths to 3 reference images
- `--output`: Output JSON file path
- `--config`: Config file (default: config/config.yaml)
- `--visualize`: Generate visualization video

### Batch Processing (Competition Dataset)

```bash
python batch_predict.py \
  --dataset /path/to/competition/dataset \
  --output submission.json \
  --visualize
```

**Dataset Structure Expected:**
```
dataset/
├── samples/
│   ├── drone_video_001/
│   │   ├── object_images/
│   │   │   ├── img_1.jpg
│   │   │   ├── img_2.jpg
│   │   │   └── img_3.jpg
│   │   └── drone_video.mp4
│   ├── drone_video_002/
│   └── ...
└── annotations/
    └── annotations.json
```

### Using Shell Script

```bash
./predict.sh path/to/video.mp4 ref1.jpg ref2.jpg ref3.jpg output.json

./predict.sh --batch /path/to/dataset submission.json
```

---

## Configuration

Edit [config/config.yaml](config/config.yaml) to adjust parameters:

```yaml
models:
  yolo:
    type: "yolov8"
    weights: "saved_models/best.pt"
    img_size: 640
    conf_threshold: 0.20
    iou_threshold: 0.45

  dinov2:
    model_name: "facebook/dinov2-small"

inference:
  matching_confidence_threshold: 0.60
  tracking_confidence_threshold: 0.40
  redetection_lost_frames: 20
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `conf_threshold` | 0.20 | YOLO confidence threshold |
| `iou_threshold` | 0.45 | NMS IoU threshold |
| `matching_confidence_threshold` | 0.60 | DINOv2 similarity threshold |
| `tracking_confidence_threshold` | 0.40 | Min confidence to stay in tracking mode |
| `redetection_lost_frames` | 20 | Frames to wait before switching to search |

---

## Output Format

### Single Video Output
```json
[
  {"frame": 370, "x1": 422, "y1": 310, "x2": 470, "y2": 355},
  {"frame": 371, "x1": 424, "y1": 312, "x2": 468, "y2": 354},
  {"frame": 372, "x1": 426, "y1": 314, "x2": 469, "y2": 356}
]
```

### Competition Submission Format
```json
[
  {
    "video_id": "drone_video_001",
    "detections": [
      {
        "bboxes": [
          {"frame": 370, "x1": 422, "y1": 310, "x2": 470, "y2": 355},
          {"frame": 371, "x1": 424, "y1": 312, "x2": 468, "y2": 354}
        ]
      }
    ]
  },
  {
    "video_id": "drone_video_002",
    "detections": []
  }
]
```

---

## API Usage

### Programmatic Access

```python
from predict import HybridPredictor

predictor = HybridPredictor(
    reference_images=['ref1.jpg', 'ref2.jpg', 'ref3.jpg'],
    config_path='config/config.yaml'
)

video_path = 'drone_video.mp4'
predictions = predictor.process_video(
    video_path=video_path,
    output_path='predictions.json',
    visualize=True
)

print(f"Detected in {len(predictions)} frames")
```

### Streaming Inference (Required for Competition)

```python
import cv2

predictor = HybridPredictor(reference_images=['ref1.jpg', 'ref2.jpg', 'ref3.jpg'])

cap = cv2.VideoCapture('drone_video.mp4')
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = predictor.predict_streaming(frame_rgb, frame_idx)

    if result:
        bbox = result['bbox']
        confidence = result['confidence']
        print(f"Frame {frame_idx}: Detected at {bbox} (conf={confidence:.3f})")

    frame_idx += 1

cap.release()
```

---

## Performance

### Expected Performance on Jetson Xavier NX (16GB)

| Mode | FPS | Description |
|------|-----|-------------|
| **Tracking (High Conf)** | 35-40 | Detection every 15 frames |
| **Tracking (Medium Conf)** | 30-35 | Detection every 10 frames |
| **Tracking (Low Conf)** | 25-30 | Detection every 5 frames |
| **Search Mode** | 20-25 | Detection every frame |

### Optimization Tips

**1. TensorRT Conversion (3-5x speedup):**
```bash
yolo export model=saved_models/best.pt format=engine half=True device=0
```

**2. Reduce Image Size:**
```yaml
yolo:
  img_size: 480  # Instead of 640
```

**3. Increase Detection Interval:**
```yaml
inference:
  detection_interval_high_conf: 20  # Instead of 15
```

---

## Troubleshooting

### Issue: YOLOv8 model not loading
**Solution:** Ensure `ultralytics` is installed:
```bash
pip install ultralytics
```

### Issue: DINOv2 out of memory
**Solution:** Reduce batch size or use CPU for feature extraction:
```yaml
dinov2:
  device: "cpu"
```

### Issue: Low ST-IoU score
**Solutions:**
1. Lower `matching_confidence_threshold` (e.g., 0.50)
2. Increase `redetection_lost_frames` (e.g., 30)
3. Fine-tune YOLO on your specific object classes

### Issue: Slow inference
**Solutions:**
1. Convert models to TensorRT
2. Increase detection intervals
3. Reduce image size
4. Use YOLOv8n instead of YOLOv8s

---

## Model Parameter Budget (Jetson Xavier NX)

| Component | Parameters | Percentage |
|-----------|------------|------------|
| YOLOv8s | 11M | 22% |
| DINOv2-small | 22M | 44% |
| ByteTrack | <1M | <2% |
| **Total** | **~33M** | **66%** |
| **Budget** | 50M | 100% |
| **Remaining** | 17M | 34% |

---

## Citation

If you use this code, please cite the original works:

```bibtex
@article{yolov8,
  title={YOLOv8: Real-Time Object Detection},
  author={Ultralytics},
  year={2023}
}

@article{dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab et al.},
  journal={arXiv preprint arXiv:2304.07193},
  year={2023}
}

@inproceedings{bytetrack,
  title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author={Zhang et al.},
  booktitle={ECCV},
  year={2022}
}
```

---

## License

MIT License

---

## Contact

For questions or issues, please open a GitHub issue or contact the development team.
