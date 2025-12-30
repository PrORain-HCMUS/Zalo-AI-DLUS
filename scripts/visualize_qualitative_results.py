import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse


def load_predictions(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    predictions = {}
    for video_data in data:
        video_id = video_data['video_id']
        bboxes = []
        if video_data['detections']:
            for det in video_data['detections']:
                bboxes.extend(det['bboxes'])
        predictions[video_id] = sorted(bboxes, key=lambda x: x['frame'])
    
    return predictions


def extract_frame(video_path, frame_number):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def draw_bbox(frame, bbox, color=(0, 255, 0), thickness=2, label=None):
    frame_vis = frame.copy()
    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
    
    cv2.rectangle(frame_vis, (x1, y1), (x2, y2), color, thickness)
    
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )
        
        cv2.rectangle(
            frame_vis,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        cv2.putText(
            frame_vis, label,
            (x1, y1 - baseline - 5),
            font, font_scale,
            (255, 255, 255),
            font_thickness
        )
    
    return frame_vis


def create_tracking_sequence(video_path, bboxes, output_path, num_frames=6):
    if len(bboxes) < num_frames:
        print(f"Warning: Only {len(bboxes)} frames available, requested {num_frames}")
        num_frames = len(bboxes)
    
    frames_vis = []
    
    for i in range(num_frames):
        bbox = bboxes[i]
        frame = extract_frame(video_path, bbox['frame'])
        
        if frame is None:
            print(f"Warning: Could not extract frame {bbox['frame']}")
            continue
        
        frame_vis = draw_bbox(
            frame, bbox,
            color=(0, 255, 0),
            label=f"Frame {bbox['frame']}"
        )
        
        frames_vis.append(frame_vis)
    
    if not frames_vis:
        print("Error: No frames extracted")
        return
    
    grid_rows = 2
    grid_cols = (num_frames + 1) // 2
    
    h, w = frames_vis[0].shape[:2]
    grid = np.zeros((h * grid_rows, w * grid_cols, 3), dtype=np.uint8)
    
    for idx, frame_vis in enumerate(frames_vis):
        row = idx // grid_cols
        col = idx % grid_cols
        grid[row*h:(row+1)*h, col*w:(col+1)*w] = frame_vis
    
    cv2.imwrite(str(output_path), grid)
    print(f"Saved tracking sequence to {output_path}")


def create_small_object_visualization(video_path, bboxes, output_path):
    small_bboxes = []
    for bbox in bboxes:
        width = bbox['x2'] - bbox['x1']
        height = bbox['y2'] - bbox['y1']
        area = width * height
        if area < 5000:
            small_bboxes.append((bbox, area))
    
    if not small_bboxes:
        print("No small objects found")
        return
    
    small_bboxes.sort(key=lambda x: x[1])
    
    num_examples = min(4, len(small_bboxes))
    frames_vis = []
    
    for i in range(num_examples):
        bbox, area = small_bboxes[i]
        frame = extract_frame(video_path, bbox['frame'])
        
        if frame is None:
            continue
        
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        
        margin = 50
        crop_x1 = max(0, x1 - margin)
        crop_y1 = max(0, y1 - margin)
        crop_x2 = min(frame.shape[1], x2 + margin)
        crop_y2 = min(frame.shape[0], y2 + margin)
        
        cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        
        adjusted_bbox = {
            'x1': x1 - crop_x1,
            'y1': y1 - crop_y1,
            'x2': x2 - crop_x1,
            'y2': y2 - crop_y1,
            'frame': bbox['frame']
        }
        
        frame_vis = draw_bbox(
            cropped, adjusted_bbox,
            color=(0, 255, 255),
            label=f"Frame {bbox['frame']} ({int(area)}px)"
        )
        
        frames_vis.append(frame_vis)
    
    if not frames_vis:
        print("Error: No frames extracted")
        return
    
    max_h = max(f.shape[0] for f in frames_vis)
    total_w = sum(f.shape[1] for f in frames_vis)
    
    grid = np.zeros((max_h, total_w, 3), dtype=np.uint8)
    
    x_offset = 0
    for frame_vis in frames_vis:
        h, w = frame_vis.shape[:2]
        grid[:h, x_offset:x_offset+w] = frame_vis
        x_offset += w
    
    cv2.imwrite(str(output_path), grid)
    print(f"Saved small object visualization to {output_path}")


def create_occlusion_visualization(video_path, bboxes, output_path):
    if len(bboxes) < 20:
        print("Not enough frames for occlusion visualization")
        return
    
    frame_gaps = []
    for i in range(len(bboxes) - 1):
        gap = bboxes[i+1]['frame'] - bboxes[i]['frame']
        if gap > 10:
            frame_gaps.append((i, gap, bboxes[i], bboxes[i+1]))
    
    if not frame_gaps:
        print("No significant occlusion gaps found")
        return
    
    frame_gaps.sort(key=lambda x: x[1], reverse=True)
    
    idx, gap, bbox_before, bbox_after = frame_gaps[0]
    
    frames_to_show = [
        (bbox_before['frame'], "Before Occlusion"),
        (bbox_before['frame'] + gap // 2, "During Occlusion (predicted)"),
        (bbox_after['frame'], "After Occlusion")
    ]
    
    frames_vis = []
    
    for frame_num, label_text in frames_to_show:
        frame = extract_frame(video_path, frame_num)
        if frame is None:
            continue
        
        if "During" in label_text:
            predicted_x1 = (bbox_before['x1'] + bbox_after['x1']) // 2
            predicted_y1 = (bbox_before['y1'] + bbox_after['y1']) // 2
            predicted_x2 = (bbox_before['x2'] + bbox_after['x2']) // 2
            predicted_y2 = (bbox_before['y2'] + bbox_after['y2']) // 2
            
            pred_bbox = {
                'x1': predicted_x1, 'y1': predicted_y1,
                'x2': predicted_x2, 'y2': predicted_y2,
                'frame': frame_num
            }
            frame_vis = draw_bbox(frame, pred_bbox, color=(255, 165, 0), label=label_text)
        else:
            bbox = bbox_before if "Before" in label_text else bbox_after
            frame_vis = draw_bbox(frame, bbox, color=(0, 255, 0), label=label_text)
        
        frames_vis.append(frame_vis)
    
    if len(frames_vis) < 3:
        print("Error: Could not extract all frames")
        return
    
    h, w = frames_vis[0].shape[:2]
    grid = np.zeros((h, w * 3, 3), dtype=np.uint8)
    
    for idx, frame_vis in enumerate(frames_vis):
        grid[:, idx*w:(idx+1)*w] = frame_vis
    
    cv2.imwrite(str(output_path), grid)
    print(f"Saved occlusion visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize qualitative results')
    
    # Get script directory and project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    parser.add_argument('--submission', type=str, 
                       default=str(project_root / 'results' / 'submission.json'),
                       help='Path to submission.json')
    parser.add_argument('--data-dir', type=str,
                       default=str(project_root / 'data' / 'public_test' / 'samples'),
                       help='Path to data directory')
    parser.add_argument('--output-dir', type=str,
                       default=str(project_root / 'assets' / 'img'),
                       help='Output directory for visualizations')
    parser.add_argument('--video-id', type=str,
                       default='BlackBox_0',
                       help='Video ID to visualize')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading predictions from {args.submission}...")
    predictions = load_predictions(args.submission)
    
    if args.video_id not in predictions:
        print(f"Error: Video {args.video_id} not found in predictions")
        print(f"Available videos: {list(predictions.keys())}")
        return
    
    video_path = Path(args.data_dir) / args.video_id / 'drone_video.mp4'
    if not video_path.exists():
        print(f"Error: Video not found at {video_path}")
        return
    
    bboxes = predictions[args.video_id]
    print(f"Found {len(bboxes)} predictions for {args.video_id}")
    
    print("\n1. Creating tracking sequence visualization...")
    create_tracking_sequence(
        video_path, bboxes,
        output_dir / f'{args.video_id}_tracking_sequence.png',
        num_frames=6
    )
    
    print("\n2. Creating small object visualization...")
    create_small_object_visualization(
        video_path, bboxes,
        output_dir / f'{args.video_id}_small_objects.png'
    )
    
    print("\n3. Creating occlusion visualization...")
    create_occlusion_visualization(
        video_path, bboxes,
        output_dir / f'{args.video_id}_occlusion.png'
    )
    
    print(f"\nAll visualizations saved to {output_dir}")


if __name__ == '__main__':
    main()
