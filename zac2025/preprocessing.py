import cv2
import numpy as np
import torch


def preprocess_reference_images(image_paths, feature_extractor):
    reference_features = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        features = feature_extractor.extract_features(img_rgb)
        reference_features.append(features)
    ref_features_tensor = torch.stack(reference_features)
    avg_ref_features = ref_features_tensor.mean(dim=0)
    return avg_ref_features, reference_features


def match_detection_with_reference(detection, frame_rgb, reference_features, feature_extractor, threshold=0.60):
    x1, y1, x2, y2 = detection['bbox']
    x1, y1, x2, y2 = max(0, x1), max(0, y1), x2, y2
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = frame_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    try:
        det_features = feature_extractor.extract_features(crop)
        similarity = feature_extractor.compute_similarity(det_features, reference_features)
        return similarity
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        return 0.0


def find_best_matching_detection(detections, frame_rgb, reference_features, feature_extractor, threshold=0.60):
    if len(detections) == 0:
        return None
    best_match = None
    best_similarity = -1
    for det in detections:
        similarity = match_detection_with_reference(det, frame_rgb, reference_features, feature_extractor, threshold)
        if similarity > best_similarity and similarity > threshold:
            best_similarity = similarity
            best_match = det
    if best_match:
        best_match['similarity'] = best_similarity
    return best_match


def bbox_to_dict(bbox, frame_idx):
    x1, y1, x2, y2 = map(int, bbox)
    return {'frame': frame_idx, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}


def format_output_for_submission(predictions):
    formatted_output = []
    for video_id, bboxes in predictions.items():
        video_pred = {'video_id': video_id, 'detections': []}
        if len(bboxes) > 0:
            video_pred['detections'].append({'bboxes': bboxes})
        formatted_output.append(video_pred)
    return formatted_output


def visualize_prediction(frame, bbox, confidence, label="Target"):
    frame_vis = frame.copy()
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    text = f"{label}: {confidence:.2f}"
    cv2.putText(frame_vis, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame_vis


def adaptive_detection_interval(confidence, high_thresh=0.80, med_thresh=0.60, low_thresh=0.40):
    if confidence > high_thresh:
        return 15
    elif confidence > med_thresh:
        return 10
    elif confidence > low_thresh:
        return 5
    else:
        return 1
