# Quick Start Guide

## Installation (5 minutes)

```bash
cd /home/lenovo/source/zac2025
source .venv/bin/activate
pip install -r requirements.txt
```

## Test Single Video (2 minutes)

```bash
python predict.py \
  --video data/sample_video.mp4 \
  --ref-images data/ref1.jpg data/ref2.jpg data/ref3.jpg \
  --output test_predictions.json \
  --visualize
```

## Process Competition Dataset

```bash
python batch_predict.py \
  --dataset /path/to/competition/dataset \
  --output submission.json
```

## What You Get

**Architecture: YOLOv8s (11M) + DINOv2-small (22M) + ByteTrack**

- Total: 33M parameters (fits Jetson Xavier NX 50M limit)
- Speed: 25-40 FPS on Jetson Xavier NX
- Target ST-IoU: 0.70-0.80

## Key Files

- [predict.py](predict.py) - Main inference with `predict_streaming()`
- [config/config.yaml](config/config.yaml) - Adjust thresholds
- [saved_models/best.pt](saved_models/best.pt) - Your YOLOv8s model

## Tuning for Better Results

### 1. Lower matching threshold (more detections)
```yaml
matching_confidence_threshold: 0.50  # Default: 0.60
```

### 2. Faster detection intervals
```yaml
detection_interval_high_conf: 20  # Default: 15
```

### 3. Different image size
```yaml
img_size: 480  # Default: 640 (faster but less accurate)
img_size: 1280  # Slower but more accurate
```

## Next Steps

1. Test on sample data
2. Tune parameters in config.yaml
3. Convert to TensorRT for Jetson deployment
4. Run on full competition dataset

## Need Help?

See [README.md](README.md) for full documentation.
