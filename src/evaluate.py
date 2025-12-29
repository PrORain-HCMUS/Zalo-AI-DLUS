#!/usr/bin/env python3
"""
Evaluation script for Zalo AI Challenge 2025 - AeroEyes
Calculates ST-IoU, Precision, Recall, and F1-Score for object detection and tracking.
"""

import json
import argparse
import numpy as np
from typing import Dict, List, Tuple
import os
from datetime import datetime


def calculate_iou(box1: Dict, box2: Dict) -> float:
    """Calculate IoU between two bounding boxes."""
    x1_max = max(box1['x1'], box2['x1'])
    y1_max = max(box1['y1'], box2['y1'])
    x2_min = min(box1['x2'], box2['x2'])
    y2_min = min(box1['y2'], box2['y2'])
    
    if x2_min <= x1_max or y2_min <= y1_max:
        return 0.0
    
    intersection = (x2_min - x1_max) * (y2_min - y1_max)
    
    area1 = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
    area2 = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def calculate_st_iou(pred_sequence: List[Dict], gt_sequence: List[Dict], 
                     iou_threshold: float = 0.5) -> float:
    """
    Calculate Spatio-Temporal IoU for a sequence of detections.
    
    Args:
        pred_sequence: List of predicted bounding boxes with frame info
        gt_sequence: List of ground truth bounding boxes with frame info
        iou_threshold: Minimum IoU threshold for spatial matching
    
    Returns:
        ST-IoU score between 0 and 1
    """
    if not pred_sequence or not gt_sequence:
        return 0.0
    
    # Create frame-indexed dictionaries for faster lookup
    pred_by_frame = {bbox['frame']: bbox for bbox in pred_sequence}
    gt_by_frame = {bbox['frame']: bbox for bbox in gt_sequence}
    
    # Find overlapping frames
    pred_frames = set(pred_by_frame.keys())
    gt_frames = set(gt_by_frame.keys())
    common_frames = pred_frames.intersection(gt_frames)
    
    if not common_frames:
        return 0.0
    
    # Calculate spatial IoU for each common frame
    spatial_ious = []
    temporal_matches = 0
    
    for frame in common_frames:
        pred_box = pred_by_frame[frame]
        gt_box = gt_by_frame[frame]
        
        spatial_iou = calculate_iou(pred_box, gt_box)
        spatial_ious.append(spatial_iou)
        
        if spatial_iou >= iou_threshold:
            temporal_matches += 1
    
    if not spatial_ious:
        return 0.0
    
    # ST-IoU combines spatial and temporal consistency
    avg_spatial_iou = np.mean(spatial_ious)
    temporal_consistency = temporal_matches / len(spatial_ious)
    
    # Weighted combination: 70% spatial, 30% temporal
    st_iou = 0.7 * avg_spatial_iou + 0.3 * temporal_consistency
    
    return st_iou


def match_detections(predictions: List[Dict], ground_truth: List[Dict], 
                    iou_threshold: float = 0.5) -> Tuple[int, int, int]:
    """
    Match predictions with ground truth and count TP, FP, FN.
    
    Args:
        predictions: List of prediction dictionaries
        ground_truth: List of ground truth dictionaries
        iou_threshold: IoU threshold for matching
    
    Returns:
        Tuple of (true_positives, false_positives, false_negatives)
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # Create dictionaries for video-wise matching
    pred_by_video = {item['video_id']: item for item in predictions}
    gt_by_video = {item['video_id']: item for item in ground_truth}
    
    # Get all video IDs
    pred_video_ids = set(pred_by_video.keys())
    gt_video_ids = set(gt_by_video.keys())
    all_video_ids = pred_video_ids.union(gt_video_ids)
    
    for video_id in all_video_ids:
        pred_item = pred_by_video.get(video_id)
        gt_item = gt_by_video.get(video_id)
        
        # If no prediction for this video
        if pred_item is None:
            if gt_item and gt_item.get('annotations') and gt_item['annotations'][0].get('bboxes'):
                false_negatives += 1
            continue
        
        # If no ground truth for this video
        if gt_item is None:
            if pred_item.get('annotations') and pred_item['annotations'][0].get('bboxes'):
                false_positives += 1
            continue
        
        # Extract bounding box sequences
        pred_bboxes = pred_item.get('annotations', [{}])[0].get('bboxes', [])
        gt_bboxes = gt_item.get('annotations', [{}])[0].get('bboxes', [])
        
        # If no detections in either
        if not pred_bboxes and not gt_bboxes:
            continue
        
        # If prediction but no ground truth
        if pred_bboxes and not gt_bboxes:
            false_positives += 1
            continue
        
        # If ground truth but no prediction
        if gt_bboxes and not pred_bboxes:
            false_negatives += 1
            continue
        
        # Calculate ST-IoU for this video sequence
        st_iou = calculate_st_iou(pred_bboxes, gt_bboxes, iou_threshold)
        
        if st_iou >= iou_threshold:
            true_positives += 1
        else:
            false_positives += 1
            false_negatives += 1
    
    return true_positives, false_positives, false_negatives


def calculate_metrics(predictions_file: str, ground_truth_file: str, 
                     iou_threshold: float = 0.5) -> Dict:
    """
    Calculate all evaluation metrics.
    
    Args:
        predictions_file: Path to predictions JSON file
        ground_truth_file: Path to ground truth JSON file
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dictionary containing all calculated metrics
    """
    # Load data
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    with open(ground_truth_file, 'r') as f:
        ground_truth = json.load(f)
    
    print(f"Loaded {len(predictions)} predictions and {len(ground_truth)} ground truth samples")
    
    # Match detections and count TP/FP/FN
    tp, fp, fn = match_detections(predictions, ground_truth, iou_threshold)
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Calculate average ST-IoU across all videos
    st_ious = []
    
    pred_by_video = {item['video_id']: item for item in predictions}
    gt_by_video = {item['video_id']: item for item in ground_truth}
    
    for video_id in set(pred_by_video.keys()).intersection(set(gt_by_video.keys())):
        pred_bboxes = pred_by_video[video_id].get('annotations', [{}])[0].get('bboxes', [])
        gt_bboxes = gt_by_video[video_id].get('annotations', [{}])[0].get('bboxes', [])
        
        if pred_bboxes and gt_bboxes:
            st_iou = calculate_st_iou(pred_bboxes, gt_bboxes, iou_threshold)
            st_ious.append(st_iou)
    
    avg_st_iou = np.mean(st_ious) if st_ious else 0.0
    
    return {
        'st_iou': avg_st_iou,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'total_predictions': len(predictions),
        'total_ground_truth': len(ground_truth),
        'iou_threshold': iou_threshold
    }


def save_results(metrics: Dict, output_file: str):
    """Save evaluation results to file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Prepare detailed results
    results = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'metrics': {
            'ST-IoU': round(metrics['st_iou'], 4),
            'Precision': round(metrics['precision'], 4),
            'Recall': round(metrics['recall'], 4),
            'F1-Score': round(metrics['f1_score'], 4)
        },
        'detailed_stats': {
            'true_positives': metrics['true_positives'],
            'false_positives': metrics['false_positives'],
            'false_negatives': metrics['false_negatives'],
            'total_predictions': metrics['total_predictions'],
            'total_ground_truth': metrics['total_ground_truth'],
            'iou_threshold': metrics['iou_threshold']
        }
    }
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate AeroEyes predictions')
    parser.add_argument('--predictions', required=True, 
                       help='Path to predictions JSON file')
    parser.add_argument('--ground-truth', required=True,
                       help='Path to ground truth JSON file')
    parser.add_argument('--output', default='results/evaluation_results.json',
                       help='Output file for results (default: results/evaluation_results.json)')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                       help='IoU threshold for matching (default: 0.5)')
    
    args = parser.parse_args()
    
    print("AeroEyes Evaluation Script")
    print("=" * 50)
    print(f"Predictions: {args.predictions}")
    print(f"Ground Truth: {args.ground_truth}")
    print(f"IoU Threshold: {args.iou_threshold}")
    print(f"Output: {args.output}")
    print()
    
    # Calculate metrics
    try:
        metrics = calculate_metrics(args.predictions, args.ground_truth, args.iou_threshold)
        
        # Display results
        print("Evaluation Results:")
        print("-" * 30)
        print(f"ST-IoU:    {metrics['st_iou']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print()
        print("Detailed Statistics:")
        print(f"True Positives:  {metrics['true_positives']}")
        print(f"False Positives: {metrics['false_positives']}")
        print(f"False Negatives: {metrics['false_negatives']}")
        print(f"Total Predictions: {metrics['total_predictions']}")
        print(f"Total Ground Truth: {metrics['total_ground_truth']}")
        
        # Save results
        save_results(metrics, args.output)
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())