import os
import cv2
import torch
import yaml
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.models import YOLOv8Detector, TPHYOLOv5Detector, DINOv2FeatureExtractor
from src.models import ByteTracker, SimpleTracker
from src.utils import (
    preprocess_reference_images,
    find_best_matching_detection,
    bbox_to_dict,
    format_output_for_submission,
    visualize_prediction,
    adaptive_detection_interval
)


class HybridPredictor:
    def __init__(self, reference_images, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        print("Initializing Hybrid Predictor (YOLOv8 + DINOv2 + Tracking)...")

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        yolo_config = self.config['models']['yolo']
        model_type = yolo_config.get('type', 'yolov8')

        if model_type == 'yolov8':
            self.detector = YOLOv8Detector(
                weights_path=yolo_config['weights'],
                img_size=yolo_config['img_size'],
                conf_thres=yolo_config['conf_threshold'],
                iou_thres=yolo_config['iou_threshold'],
                device=self.device
            )
        else:
            self.detector = TPHYOLOv5Detector(
                weights_path=yolo_config['weights'],
                img_size=yolo_config['img_size'],
                conf_thres=yolo_config['conf_threshold'],
                iou_thres=yolo_config['iou_threshold'],
                device=self.device
            )

        self.feature_extractor = DINOv2FeatureExtractor(
            model_name=self.config['models']['dinov2']['model_name'],
            device=self.device
        )

        print("Processing reference images...")
        self.ref_features_avg, self.ref_features_list = preprocess_reference_images(
            reference_images, self.feature_extractor
        )
        print(f"Reference features computed: {len(self.ref_features_list)} images")

        self.tracker = ByteTracker(max_age=30, min_hits=2, iou_threshold=0.3)
        self.simple_tracker = SimpleTracker()

        self.mode = "search"
        self.lost_frames = 0
        self.detection_interval = self.config['inference']['detection_interval_search']
        self.frame_since_last_detection = 0

        self.tracking_conf_threshold = self.config['inference']['tracking_confidence_threshold']
        self.matching_threshold = self.config['inference']['matching_confidence_threshold']
        self.redetection_lost_frames = self.config['inference']['redetection_lost_frames']

        self.predictions_buffer = []

        print("Hybrid Predictor initialized successfully!")

    def predict_streaming(self, frame_rgb_np, frame_idx):
        self.frame_since_last_detection += 1

        should_detect = (
            self.mode == "search" or
            self.frame_since_last_detection >= self.detection_interval
        )

        current_bbox = None
        current_confidence = 0.0

        if should_detect:
            matched_detection = None
            all_detections = self.detector.detect(frame_rgb_np)

            if len(all_detections) > 0:
                matched_detection = find_best_matching_detection(
                    all_detections,
                    frame_rgb_np,
                    self.ref_features_avg,
                    self.feature_extractor,
                    threshold=self.matching_threshold
                )

                if matched_detection:
                    matched_detection['frame_idx'] = frame_idx
                    tracked_objects = self.tracker.update([matched_detection])

                    if self.mode == "search":
                        self.mode = "tracking"
                        self.lost_frames = 0
                        print(f"Frame {frame_idx}: Target found! Switching to tracking mode "
                              f"(similarity={matched_detection.get('similarity', 0):.3f})")

                    self.simple_tracker.update(
                        matched_detection['bbox'],
                        matched_detection.get('similarity', matched_detection['confidence'])
                    )

                    current_bbox = matched_detection['bbox']
                    current_confidence = matched_detection.get('similarity', matched_detection['confidence'])

                    self.frame_since_last_detection = 0
                else:
                    tracked_objects = self.tracker.update([])

            else:
                tracked_objects = self.tracker.update([])

            if not matched_detection and len(tracked_objects) == 0:
                self.lost_frames += 1

                if self.lost_frames > self.redetection_lost_frames and self.mode == "tracking":
                    self.mode = "search"
                    self.detection_interval = self.config['inference']['detection_interval_search']
                    self.simple_tracker.reset()
                    print(f"Frame {frame_idx}: Tracking lost, switching to search mode")

        else:
            best_track = self.tracker.get_best_track()
            if best_track:
                current_bbox = best_track['bbox']
                current_confidence = best_track['confidence']
            else:
                predicted = self.simple_tracker.predict()
                if predicted:
                    current_bbox = predicted['bbox']
                    current_confidence = predicted['confidence']
                else:
                    self.lost_frames += 1

                    if self.lost_frames > self.redetection_lost_frames and self.mode == "tracking":
                        self.mode = "search"
                        self.detection_interval = self.config['inference']['detection_interval_search']
                        print(f"Frame {frame_idx}: Tracking lost, switching to search mode")

        if self.mode == "tracking" and current_confidence > 0:
            self.detection_interval = adaptive_detection_interval(
                current_confidence,
                high_thresh=self.config['inference']['tracking_high_conf'],
                med_thresh=self.config['inference']['tracking_med_conf'],
                low_thresh=self.config['inference']['tracking_low_conf']
            )

        if current_bbox is not None and current_confidence > 0.3:
            return {
                'bbox': current_bbox,
                'confidence': current_confidence,
                'frame_idx': frame_idx
            }

        return None

    def process_video(self, video_path, output_path=None, visualize=False):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"Processing video: {video_path}")
        print(f"Total frames: {total_frames}, FPS: {fps}")

        predictions = []
        frame_idx = 0

        if visualize:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = cv2.VideoWriter(
                output_path.replace('.json', '_vis.mp4'),
                fourcc, fps,
                (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            )

        with tqdm(total=total_frames, desc="Processing") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                result = self.predict_streaming(frame_rgb, frame_idx)

                if result:
                    bbox_dict = bbox_to_dict(result['bbox'], frame_idx)
                    predictions.append(bbox_dict)

                    if visualize:
                        frame_vis = visualize_prediction(
                            frame, result['bbox'], result['confidence']
                        )
                        out_video.write(frame_vis)
                else:
                    if visualize:
                        out_video.write(frame)

                frame_idx += 1
                pbar.update(1)

        cap.release()
        if visualize:
            out_video.release()

        print(f"Processed {frame_idx} frames, detected target in {len(predictions)} frames")

        if output_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w+') as f:
                json.dump(predictions, f, indent=2)
            print(f"Predictions saved to: {output_path}")

        return predictions


def main():
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

    predictor = HybridPredictor(
        reference_images=args.ref_images,
        config_path=args.config
    )

    predictions = predictor.process_video(
        video_path=args.video,
        output_path=args.output,
        visualize=args.visualize
    )

    print(f"\nPrediction complete!")
    print(f"Total detections: {len(predictions)}")


if __name__ == '__main__':
    main()
