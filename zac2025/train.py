import torch
import yaml
from pathlib import Path
from ultralytics import YOLO


def train_yolo(data_yaml, model='yolov8s.pt', epochs=100, img_size=640, batch_size=16):
    model = YOLO(model)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=0 if torch.cuda.is_available() else 'cpu',
        project='runs/train',
        name='yolov8s_aeroeyes',
        exist_ok=True
    )
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train YOLOv8 for AeroEyes')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml')
    parser.add_argument('--model', type=str, default='yolov8s.pt', help='Model to use')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    args = parser.parse_args()
    train_yolo(
        data_yaml=args.data,
        model=args.model,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()
