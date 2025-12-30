import torch
import yaml
from pathlib import Path
from ultralytics import YOLO


def train_yolo(data_yaml, model='yolo11s.pt', epochs=100, img_size=640, batch_size=16, project='runs/train', name='yolo11s_aeroeyes'):
    """
    Train YOLO11 model for AeroEyes object detection.
    
    Args:
        data_yaml: Path to data.yaml configuration file
        model: Pretrained model to start from
        epochs: Number of training epochs
        img_size: Input image size
        batch_size: Batch size for training
        project: Project directory for saving results
        name: Experiment name
    
    Returns:
        Training results
    """
    model = YOLO(model)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=0 if torch.cuda.is_available() else 'cpu',
        project=project,
        name=name,
        exist_ok=True
    )
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train YOLO11 for AeroEyes')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml')
    parser.add_argument('--model', type=str, default='yolo11s.pt', help='Model to use')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--project', type=str, default='runs/train', help='Project directory')
    parser.add_argument('--name', type=str, default='yolo11s_aeroeyes', help='Experiment name')
    
    args = parser.parse_args()
    
    train_yolo(
        data_yaml=args.data,
        model=args.model,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch_size,
        project=args.project,
        name=args.name
    )


if __name__ == '__main__':
    main()
