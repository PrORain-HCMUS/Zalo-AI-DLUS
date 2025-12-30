import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import json
import os
import torch.nn.functional as F
import timm
from ultralytics import YOLO
import yaml
from tqdm import tqdm
from src.models.siamese import Backbone, FPNWrapper, AttentionPooling, PrototypeExtractor


class JsonPrototypePipeline:
    def __init__(self,  yolo_model, extractor, transform, threshold=0.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        print(f"Initializing Pipeline on {self.device}...")

        # 1. Load Models
        self.detector = yolo_model
        self.extractor = extractor.to(self.device)
        self.extractor.eval()

        # 2. Preprocessing
        self.transform = transform

        # 3. Reference Storage
        # Store as a tensor matrix for vectorized comparison [N_refs, Emb_Dim]
        self.ref_tensor = None

    def assign_ref_img(self, ref_image_paths):
        self.process_reference_images(ref_image_paths)

    def get_embedding_batch(self, img_tensors):
        """
        Runs the extractor on a batch of images.
        Input: Tensor of shape [Batch_Size, C, H, W]
        Output: Tensor of shape [Batch_Size, Embed_Dim] (Normalized)
        """
        with torch.inference_mode():
            features = self.extractor(img_tensors)
            # Flatten if necessary (depending on model architecture)
            if len(features.shape) > 2:
                features = features.flatten(start_dim=1)
            # Normalize features for Cosine Similarity
            features = F.normalize(features, p=2, dim=1)
            return features

    def process_reference_images(self, paths):
        ref_embeddings = []
        for i, path in enumerate(paths):
            try:
                img = Image.open(path).convert('RGB')
                # Transform and add batch dimension
                img_t = self.transform(img).unsqueeze(0).to(self.device)

                # Extract and normalize
                with torch.inference_mode():
                    emb = self.extractor(img_t).flatten(start_dim=1)
                    emb = F.normalize(emb, p=2, dim=1)

                ref_embeddings.append(emb)
            except Exception as e:
                print(f"Error loading {path}: {e}")

        if ref_embeddings:
            # Stack into a single matrix: [N_refs, Embed_Dim]
            self.ref_tensor = torch.cat(ref_embeddings, dim=0)

    def process_batch(self, video_path, video_name, output_json='predictions.json'):
        if self.ref_tensor is None:
            print("No reference images assigned. Skipping.")
            return


        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0
        object_tracks = []
        
        with tqdm(total=total_frames, desc=f"Processing {video_name}", unit="frames") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                h, w, _ = frame.shape

                # YOLO Inference
                results = self.detector(frame, verbose=False, conf=0.1)

                # Collect crops for batch processing
                crops = []
                coords = []

                for result in results:
                    if result.boxes is None: continue

                    # Get boxes in one go (on CPU for slicing)
                    boxes = result.boxes.xyxy.cpu().numpy().astype(int)

                    for box in boxes:
                        x1, y1, x2, y2 = box

                        # Clamp coordinates
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)

                        # Basic size check
                        if x2 - x1 < 2 or y2 - y1 < 2:
                            continue

                        # Crop (Keep numpy for now)
                        obj_crop = frame[y1:y2, x1:x2]

                        # Convert to PIL for Transform (to match original logic)
                        # Note: This part is CPU bound, difficult to optimize without changing transform logic completely
                        obj_crop_pil = Image.fromarray(cv2.cvtColor(obj_crop, cv2.COLOR_BGR2RGB))

                        crops.append(self.transform(obj_crop_pil))
                        coords.append((x1, y1, x2, y2))

                # Batch Inference
                if crops:
                    batch_t = torch.stack(crops).to(self.device)

                    batch_emb = self.get_embedding_batch(batch_t)

                    sim_matrix = torch.mm(batch_emb, self.ref_tensor.T)

                    max_vals, max_indices = torch.max(sim_matrix, dim=1)
                    max_vals_cpu = max_vals.cpu().numpy()


                    for idx, score in enumerate(max_vals_cpu):

                        # if score > self.threshold: <--- khỏi filter thì tốt hơn
                            x1, y1, x2, y2 = coords[idx]
                            bbox_entry = {
                                "frame": frame_idx,
                                "x1": int(x1), "y1": int(y1),
                                "x2": int(x2), "y2": int(y2)
                            }
                            object_tracks.append(bbox_entry)

                frame_idx += 1
                pbar.update(1)
        
        cap.release()


        video_detections_dict = {}

        for bboxes in object_tracks:
            if bboxes:
                if "bboxes" not in video_detections_dict:
                    video_detections_dict["bboxes"] = []
                video_detections_dict["bboxes"].append(bboxes)

        new_entry = {
            "video_id": video_name,
            "detections": [video_detections_dict]
        }


        with open(output_json, 'w') as f:
            f.write('[\n')
            for i, bbox_dict in enumerate(video_detections_dict['bboxes']):
                json_line = json.dumps(bbox_dict, separators=(',', ':'))
                if i < len(video_detections_dict['bboxes']) - 1:
                    f.write(f'{json_line},\n')
                else:
                    f.write(f'{json_line}\n')
            f.write(']')
        print(f"Predictions on {video_name} saved to: {output_json}")



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Hybrid Predictor for AeroEyes')
    parser.add_argument('--video', type=str, required=True, help='Path to drone video')
    parser.add_argument('--ref-images', type=str, nargs=3, required=True,
                        help='Paths to 3 reference images')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--output', type=str, default='predictions.json',
                        help='Output JSON file path')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization video')
    args = parser.parse_args()
    config_path = args.config
    with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    detection_model_path = config['models']['yolo']['weights']
    comparing_model_path = config['models']['siamese']['weights']
    device = "cuda" if torch.cuda.is_available() else "cpu"

    yolo_model = YOLO(detection_model_path)

    model = timm.create_model(
      "hf_hub:timm/mobilenetv4_conv_medium.e500_r224_in1k",
        pretrained=True,
        features_only=True,
        out_indices=(1, 2, 4)
    ).to(device)
    model.eval()
    backbone = Backbone(model)
    fpn = FPNWrapper(backbone)
    att = AttentionPooling()
    prototypeExtractor = PrototypeExtractor(fpn, att)

    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    if os.path.exists(comparing_model_path):
        print(f"Loading checkpoint from {comparing_model_path}...")
        checkpoint = torch.load(comparing_model_path, map_location=device)

        # A. Load Model Weights
        prototypeExtractor.load_state_dict(checkpoint['model_state_dict'])
        prototypeExtractor.to(device)
    prototypeExtractor.eval()

    pipeline = JsonPrototypePipeline(
        yolo_model = yolo_model,
        extractor = prototypeExtractor,
        transform = transforms,
        threshold=config['models']['siamese']['threshold'])
    
    test_data_path = args.video
    ref_images = args.ref_images
    pipeline.assign_ref_img(ref_images)
    pipeline.process_batch(video_path=test_data_path, video_name = args.video.split("/")[4], output_json=args.output)

