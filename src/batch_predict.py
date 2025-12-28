import os
import json
from pathlib import Path
from tqdm import tqdm
from src.predict import HybridPredictor


def process_competition_dataset(dataset_root, output_file, config_path='config/config.yaml', visualize=False):
    """
    Process entire competition dataset with multiple videos.
    
    Args:
        dataset_root: Root directory containing samples/ and annotations/
        output_file: Output JSON file for submission
        config_path: Path to configuration file
        visualize: Whether to generate visualization videos
    
    Returns:
        List of predictions for all videos
    """
    samples_dir = Path(dataset_root) / 'samples'

    if not samples_dir.exists():
        raise ValueError(f"Samples directory not found: {samples_dir}")

    video_dirs = sorted([d for d in samples_dir.iterdir() if d.is_dir()])

    print(f"Found {len(video_dirs)} videos to process")

    all_predictions = []

    for video_dir in tqdm(video_dirs, desc="Processing videos"):
        video_id = video_dir.name

        object_images_dir = video_dir / 'object_images'
        video_file = video_dir / 'drone_video.mp4'

        if not video_file.exists():
            print(f"Warning: Video not found for {video_id}, skipping...")
            continue

        if not object_images_dir.exists():
            print(f"Warning: Object images not found for {video_id}, skipping...")
            continue

        ref_images = sorted(list(object_images_dir.glob('*.jpg')) + list(object_images_dir.glob('*.png')))

        if len(ref_images) < 3:
            print(f"Warning: Less than 3 reference images found for {video_id}, skipping...")
            continue

        ref_images = [str(img) for img in ref_images[:3]]

        print(f"\n{'='*60}")
        print(f"Processing: {video_id}")
        print(f"Video: {video_file}")
        print(f"Reference images: {len(ref_images)}")
        print(f"{'='*60}")

        try:
            predictor = HybridPredictor(
                reference_images=ref_images,
                config_path=config_path
            )

            output_json = f"results/{video_id}_predictions.json"
            os.makedirs('results', exist_ok=True)

            predictions = predictor.process_video(
                video_path=str(video_file),
                output_path=output_json,
                visualize=visualize
            )

            video_prediction = {
                'video_id': video_id,
                'detections': []
            }

            if len(predictions) > 0:
                video_prediction['detections'].append({
                    'bboxes': predictions
                })

            all_predictions.append(video_prediction)

            print(f"✓ {video_id}: {len(predictions)} frames detected")

        except Exception as e:
            print(f"✗ Error processing {video_id}: {e}")
            all_predictions.append({
                'video_id': video_id,
                'detections': []
            })

    with open(output_file, 'w') as f:
        json.dump(all_predictions, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Batch processing complete!")
    print(f"Total videos processed: {len(all_predictions)}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")

    return all_predictions


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Batch process competition dataset')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to competition dataset root (contains samples/ and annotations/)')
    parser.add_argument('--output', type=str, default='submission.json',
                        help='Output submission JSON file')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization videos')

    args = parser.parse_args()

    process_competition_dataset(
        dataset_root=args.dataset,
        output_file=args.output,
        config_path=args.config,
        visualize=args.visualize
    )


if __name__ == '__main__':
    main()
