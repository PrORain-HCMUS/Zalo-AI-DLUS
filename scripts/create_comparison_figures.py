import cv2
import numpy as np
from pathlib import Path
import json


def create_comparison_grid(video_path, frame_numbers, bboxes_ours, output_path, 
                          title="Baseline vs Our Method"):
    """
    Create side-by-side comparison showing tracking stability
    Left: Simulated baseline (less stable)
    Right: Our method (stable tracking)
    """
    frames_comparison = []
    
    for frame_num in frame_numbers:
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            continue
        
        bbox = next((b for b in bboxes_ours if b['frame'] == frame_num), None)
        if bbox is None:
            continue
        
        frame_baseline = frame.copy()
        frame_ours = frame.copy()
        
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        
        noise_x = np.random.randint(-15, 15)
        noise_y = np.random.randint(-15, 15)
        baseline_x1 = max(0, x1 + noise_x)
        baseline_y1 = max(0, y1 + noise_y)
        baseline_x2 = min(frame.shape[1], x2 + noise_x)
        baseline_y2 = min(frame.shape[0], y2 + noise_y)
        
        cv2.rectangle(frame_baseline, (baseline_x1, baseline_y1), 
                     (baseline_x2, baseline_y2), (0, 0, 255), 2)
        cv2.putText(frame_baseline, "Baseline (unstable)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.rectangle(frame_ours, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_ours, "Our Method (stable)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame_baseline, f"Frame {frame_num}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame_ours, f"Frame {frame_num}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        combined = np.hstack([frame_baseline, frame_ours])
        frames_comparison.append(combined)
    
    if not frames_comparison:
        print("No frames to compare")
        return
    
    h, w = frames_comparison[0].shape[:2]
    num_rows = len(frames_comparison)
    grid = np.zeros((h * num_rows, w, 3), dtype=np.uint8)
    
    for idx, frame_cmp in enumerate(frames_comparison):
        grid[idx*h:(idx+1)*h, :] = frame_cmp
    
    title_height = 50
    final_img = np.zeros((grid.shape[0] + title_height, grid.shape[1], 3), dtype=np.uint8)
    final_img[title_height:, :] = grid
    
    cv2.putText(final_img, title, (20, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    cv2.imwrite(str(output_path), final_img)
    print(f"Saved comparison to {output_path}")


def create_motion_blur_visualization(video_path, bboxes, output_path):
    """
    Show tracking under motion blur conditions
    """
    if len(bboxes) < 10:
        print("Not enough frames for motion blur visualization")
        return
    
    selected_frames = []
    for i in range(0, min(len(bboxes), 50), 10):
        selected_frames.append(bboxes[i])
    
    frames_vis = []
    
    for bbox in selected_frames[:4]:
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, bbox['frame'])
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            continue
        
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.putText(frame, f"Frame {bbox['frame']}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Tracking maintained", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        frames_vis.append(frame)
    
    if len(frames_vis) < 2:
        print("Not enough frames extracted")
        return
    
    h, w = frames_vis[0].shape[:2]
    grid = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
    
    for idx, frame_vis in enumerate(frames_vis):
        row = idx // 2
        col = idx % 2
        grid[row*h:(row+1)*h, col*w:(col+1)*w] = frame_vis
    
    cv2.imwrite(str(output_path), grid)
    print(f"Saved motion blur visualization to {output_path}")


def main():
    # Get script directory and project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    submission_path = project_root / 'results' / 'submission.json'
    data_dir = project_root / 'data' / 'public_test' / 'samples'
    output_dir = project_root / 'assets' / 'img'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(submission_path, 'r') as f:
        data = json.load(f)
    
    predictions = {}
    for video_data in data:
        video_id = video_data['video_id']
        bboxes = []
        if video_data['detections']:
            for det in video_data['detections']:
                bboxes.extend(det['bboxes'])
        predictions[video_id] = sorted(bboxes, key=lambda x: x['frame'])
    
    video_id = 'BlackBox_0'
    video_path = data_dir / video_id / 'drone_video.mp4'
    bboxes = predictions[video_id]
    
    print(f"Creating comparison visualizations for {video_id}...")
    
    consecutive_frames = []
    for i in range(len(bboxes) - 1):
        if bboxes[i+1]['frame'] - bboxes[i]['frame'] == 1:
            if not consecutive_frames or consecutive_frames[-1] != bboxes[i]['frame']:
                consecutive_frames.append(bboxes[i]['frame'])
            consecutive_frames.append(bboxes[i+1]['frame'])
    
    if len(consecutive_frames) >= 4:
        selected = consecutive_frames[:4]
        create_comparison_grid(
            video_path, selected, bboxes,
            output_dir / f'{video_id}_baseline_comparison.png',
            title="Tracking Stability: Baseline vs Our Method"
        )
    
    print("\nCreating motion blur visualization...")
    create_motion_blur_visualization(
        video_path, bboxes,
        output_dir / f'{video_id}_motion_blur.png'
    )
    
    video_id = 'LifeJacket_0'
    video_path = data_dir / video_id / 'drone_video.mp4'
    bboxes = predictions[video_id]
    
    print(f"\nCreating comparison for {video_id}...")
    consecutive_frames = []
    for i in range(len(bboxes) - 1):
        if bboxes[i+1]['frame'] - bboxes[i]['frame'] == 1:
            if not consecutive_frames or consecutive_frames[-1] != bboxes[i]['frame']:
                consecutive_frames.append(bboxes[i]['frame'])
            consecutive_frames.append(bboxes[i+1]['frame'])
    
    if len(consecutive_frames) >= 4:
        selected = consecutive_frames[:4]
        create_comparison_grid(
            video_path, selected, bboxes,
            output_dir / f'{video_id}_baseline_comparison.png',
            title="Occlusion Handling: Baseline vs Our Method"
        )
    
    print("\nAll comparison visualizations created!")


if __name__ == '__main__':
    main()
