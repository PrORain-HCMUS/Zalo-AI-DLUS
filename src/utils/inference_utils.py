import cv2
import numpy as np
import torch

def xywh_norm_to_xyxy_abs(bbox_norm, img_width = 1024, img_height=576):
    # Unpack normalized center coordinates and dimensions
    norm_cx, norm_cy, norm_w, norm_h = bbox_norm

    # Convert to absolute pixel values
    abs_cx = round(norm_cx * img_width, 3)
    abs_cy = round(norm_cy * img_height, 3)
    abs_w = round(norm_w * img_width, 3)
    abs_h = round(norm_h * img_height, 3)

    # Convert from center (x,y) to top-left (x,y)
    abs_x = abs_cx - (abs_w / 2)
    abs_y = abs_cy - (abs_h / 2)

    return [abs_x, abs_y, abs_x + abs_w, abs_y + abs_h]

def str2list(s: str, round_values: bool = False):
    tokens = s.strip().split()
    return [float(t) for t in tokens]

def calculate_iou(boxA, boxB):
    # ... (implementation from above) ...
    inter_x1 = max(boxA[0], boxB[0])
    inter_y1 = max(boxA[1], boxB[1])
    inter_x2 = min(boxA[2], boxB[2])
    inter_y2 = min(boxA[3], boxB[3])
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    if inter_area == 0: return 0.0
    area_A = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    area_B = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = area_A + area_B - inter_area
    return inter_area / union_area

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
        similarity = match_detection_with_reference(
            det, frame_rgb, reference_features, feature_extractor, threshold
        )

        if similarity > best_similarity and similarity > threshold:
            best_similarity = similarity
            best_match = det

    if best_match:
        best_match['similarity'] = best_similarity

    return best_match


def bbox_to_dict(bbox, frame_idx):
    x1, y1, x2, y2 = map(int, bbox)
    return {
        'frame': frame_idx,
        'x1': x1,
        'y1': y1,
        'x2': x2,
        'y2': y2
    }


def format_output_for_submission(predictions):
    formatted_output = []

    for video_id, bboxes in predictions.items():
        video_pred = {
            'video_id': video_id,
            'detections': []
        }

        if len(bboxes) > 0:
            video_pred['detections'].append({
                'bboxes': bboxes
            })

        formatted_output.append(video_pred)

    return formatted_output


def visualize_prediction(frame, bbox, confidence, label="Target"):
    frame_vis = frame.copy()

    x1, y1, x2, y2 = map(int, bbox)

    cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    text = f"{label}: {confidence:.2f}"
    cv2.putText(frame_vis, text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
