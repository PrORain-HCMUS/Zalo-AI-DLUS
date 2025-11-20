import sys
import torch
from pathlib import Path


class YOLOv8Detector:
    def __init__(self, weights_path, img_size=640, conf_thres=0.20, iou_thres=0.45, device='cuda'):
        from ultralytics import YOLO

        self.device = device if torch.cuda.is_available() else 'cpu'
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        print(f"Loading YOLOv8 model from {weights_path}...")
        self.model = YOLO(weights_path)
        self.model.to(self.device)

        if device == 'cuda' and torch.cuda.is_available():
            self.fp16 = True
        else:
            self.fp16 = False

        try:
            self.names = self.model.names
        except:
            self.names = {}

        print(f"YOLOv8 loaded successfully. Device: {self.device}")

    def detect(self, img_rgb):
        results = self.model(
            img_rgb,
            imgsz=self.img_size,
            conf=self.conf_thres,
            iou=self.iou_thres,
            verbose=False,
            half=self.fp16
        )

        detections = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())

                x1, y1, x2, y2 = map(int, xyxy)

                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'class': cls,
                    'class_name': self.names.get(cls, f'class_{cls}')
                })

        return detections


sys.path.insert(0, str(Path(__file__).parent.parent / 'tph-yolov5'))

try:
    from models.experimental import attempt_load
    from utils.general import non_max_suppression, scale_coords
    TPH_AVAILABLE = True
except ImportError:
    TPH_AVAILABLE = False
    print("Warning: TPH-YOLOv5 not available. Use YOLOv8 instead.")


class TPHYOLOv5Detector:
    def __init__(self, weights_path, img_size=1280, conf_thres=0.20, iou_thres=0.45, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        print(f"Loading TPH-YOLOv5 model from {weights_path}...")
        self.model = attempt_load(weights_path, map_location=self.device)
        self.model.eval()

        if device == 'cuda' and torch.cuda.is_available():
            self.model.half()
            self.fp16 = True
        else:
            self.fp16 = False

        self.stride = int(self.model.stride.max())
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        print(f"TPH-YOLOv5 loaded successfully. Classes: {self.names}")

    def preprocess(self, img):
        import cv2
        import numpy as np

        img_resized = cv2.resize(img, (self.img_size, self.img_size))

        img_tensor = img_resized.transpose((2, 0, 1))[::-1]
        img_tensor = np.ascontiguousarray(img_tensor)
        img_tensor = torch.from_numpy(img_tensor).to(self.device)
        img_tensor = img_tensor.half() if self.fp16 else img_tensor.float()
        img_tensor /= 255.0

        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        return img_tensor, img_resized.shape

    def detect(self, img_rgb):
        img_tensor, resized_shape = self.preprocess(img_rgb)

        with torch.no_grad():
            pred = self.model(img_tensor)[0]

        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None, agnostic=False)

        detections = []
        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img_rgb.shape).round()

                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(int, xyxy)
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(conf),
                        'class': int(cls),
                        'class_name': self.names[int(cls)]
                    })

        return detections


class DINOv2FeatureExtractor:
    def __init__(self, model_name='facebook/dinov2-small', device='cuda'):
        from transformers import AutoImageProcessor, AutoModel

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Loading DINOv2 model: {model_name}...")

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        if device == 'cuda' and torch.cuda.is_available():
            self.model.half()
            self.fp16 = True
        else:
            self.fp16 = False

        print(f"DINOv2 loaded successfully on {self.device}")

    def extract_features(self, img_rgb):
        import torch.nn.functional as F
        from PIL import Image

        if isinstance(img_rgb, Image.Image):
            pil_img = img_rgb
        else:
            import cv2
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR) if img_rgb.shape[2] == 3 else img_rgb
            pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        inputs = self.processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if self.fp16:
            inputs['pixel_values'] = inputs['pixel_values'].half()

        with torch.no_grad():
            outputs = self.model(**inputs)

        features = outputs.last_hidden_state[:, 0, :]
        features = F.normalize(features, p=2, dim=1)

        return features.squeeze(0)

    def compute_similarity(self, features1, features2):
        return torch.nn.functional.cosine_similarity(
            features1.unsqueeze(0),
            features2.unsqueeze(0),
            dim=1
        ).item()
