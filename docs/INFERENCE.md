# Inference Guide

## Overview

This guide explains how to run inference on drone videos to detect and track target objects.

## Prerequisites

- Trained model weights at `checkpoints/best.pt`
- Virtual environment activated
- Test video and reference images

## Single Video Inference

### Basic Usage

```bash
python src/predict.py \
  --video path/to/video.mp4 \
  --ref-images ref1.jpg ref2.jpg ref3.jpg \
  --output predictions.json
```

### With Visualization

```bash
python src/predict.py \
  --video path/to/video.mp4 \
  --ref-images ref1.jpg ref2.jpg ref3.jpg \
  --output predictions.json \
  --visualize
```

This generates:
- `predictions.json` - Detection results
- `predictions_vis.mp4` - Visualization video

### Custom Configuration

```bash
python src/predict.py \
  --video path/to/video.mp4 \
  --ref-images ref1.jpg ref2.jpg ref3.jpg \
  --config config/config.yaml \
  --output predictions.json
```

## Batch Processing

Process entire competition dataset:

```bash
python src/batch_predict.py \
  --dataset data/test \
  --output submission.json
```

With visualization:

```bash
python src/batch_predict.py \
  --dataset data/test \
  --output submission.json \
  --visualize
```

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

## Configuration Tuning

Edit `config/config.yaml` to adjust detection parameters:

### For Higher Recall (Detect More)

```yaml
models:
  yolo:
    conf_threshold: 0.15  # Lower threshold

inference:
  matching_confidence_threshold: 0.50  # Lower similarity threshold
```

### For Higher Precision (Fewer False Positives)

```yaml
models:
  yolo:
    conf_threshold: 0.30  # Higher threshold

inference:
  matching_confidence_threshold: 0.70  # Higher similarity threshold
```

### For Faster Inference

```yaml
models:
  yolo:
    img_size: 480  # Smaller image size

inference:
  detection_interval_high_conf: 20  # Less frequent detection
```

## Performance Optimization

### 1. TensorRT Conversion (3-5x speedup)

```bash
yolo export model=checkpoints/best.pt format=engine half=True device=0
```

Update config:
```yaml
models:
  yolo:
    weights: "checkpoints/best.engine"
```

### 2. Reduce Image Size

```yaml
models:
  yolo:
    img_size: 480  # Instead of 640
```

### 3. Adjust Detection Intervals

```yaml
inference:
  detection_interval_high_conf: 20  # Instead of 15
  detection_interval_med_conf: 15   # Instead of 10
```

## Programmatic Usage

```python
from src.predict import HybridPredictor

# Initialize predictor
predictor = HybridPredictor(
    reference_images=['ref1.jpg', 'ref2.jpg', 'ref3.jpg'],
    config_path='config/config.yaml'
)

# Process video
predictions = predictor.process_video(
    video_path='video.mp4',
    output_path='predictions.json',
    visualize=True
)

print(f"Detected in {len(predictions)} frames")
```

## Troubleshooting

### No Detections

1. Lower confidence thresholds
2. Check reference images quality
3. Verify model weights are correct

### Too Many False Positives

1. Increase confidence thresholds
2. Use higher similarity threshold
3. Improve reference images

### Slow Inference

1. Convert to TensorRT
2. Reduce image size
3. Increase detection intervals
4. Use GPU if available

## Tips

1. **Reference images**: Use clear, well-lit images of the target
2. **Confidence tuning**: Start with default values, adjust based on results
3. **Visualization**: Always generate visualization for first run to verify
4. **Batch processing**: Use for competition submission
5. **GPU**: Ensure CUDA is available for faster inference
