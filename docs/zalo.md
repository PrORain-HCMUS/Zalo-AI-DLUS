# Zalo AI Challenge — Spatio-Temporal Object Localization

## 1. Qualification Round

The evaluation uses a **3-D Spatio–Temporal Intersection-over-Union (ST-IoU)** metric that jointly measures *when* and *where* the target object is correctly detected in the video. Unlike traditional metrics that treat temporal and spatial accuracy separately, ST-IoU considers them as one continuous space–time volume.

A detection only receives credit if both the **timing** and the **bounding boxes** align correctly with the ground truth.

---

## 1.1 Definition

For each video, let:

- **Ground-truth bounding boxes:** \( B_f \) for frame \( f \)
- **Predicted bounding boxes:** \( B_f' \) at frame \( f \)

The **Spatio–Temporal IoU (ST-IoU)** is computed as:

\[
STIoU = \frac{\sum\limits_{f \in intersection} IoU(B_f, B_f')}
{\sum\limits_{f \in union} 1}
\]

Where:

- **intersection** — overlapping frames between predicted and ground-truth  
- **IoU(\(B_f, B_f'\))** — spatial IoU of bounding boxes at frame \( f \)  
- **union** — all frames that belong to either the ground-truth or the predicted  

---

## 1.2 Scoring and Aggregation

For each video, the final score is the ST-IoU value between predicted and ground-truth spatio–temporal volumes.  
The leaderboard score is the mean ST-IoU across all evaluation videos:

\[
Final\ Score = \frac{1}{N} \sum_{i=1}^{N} STIoU_{video_i}
\]

---

## 2. Final Round

The top 5 teams from the qualification phase will advance to the on-site final in Ho Chi Minh City. Each team will deploy their model on a real drone equipped with NVIDIA Jetson hardware. The drone must autonomously search for unknown objects in a physical terrain setup.

Teams will be evaluated based on:

- Detection accuracy (temporal + spatial)  
- Search efficiency (time to locate the object)  
- Real-time performance (inference speed and stability)

---

# Challenge Description

In emergency and disaster response scenarios, autonomous drones play a crucial role in locating missing persons or critical objects in challenging environments such as flooded zones, forests, or post-storm areas.

This challenge encourages participants to design AI models capable of searching for and localizing a specific object from the drone, based on limited reference images.

**Your mission:**  
Build a perception system that can determine **when and where** a given target object appears in drone-captured footage — simulating a real-world search-and-rescue mission.

The qualification round is conducted on pre-recorded drone videos provided by the organizers:

- You are provided with **three images** of a target object (backpack, person, bicycle, laptop...)  
- And a **drone video** scanning an area  
- Your task is to **predict bounding boxes** of the object in each detected frame

This is a **spatio-temporal localization** task requiring recognition under varying scales, viewpoints, and occlusions.

In the final round, the same algorithms must run **in real time** on drones (Jetson hardware).

---

# Data

## 1. Directory Structure

All data are provided in a single folder containing reference images, drone videos, and ground-truth annotations.

dataset/
├── samples/
│   ├── drone_video_001/
│   │   ├── object_images/
│   │   │   ├── img_1.jpg
│   │   │   ├── img_2.jpg
│   │   │   └── img_3.jpg
│   │   └── drone_video.mp4
│   ├── drone_video_002/
│   │   ├── object_images/
│   │   │   ├── img_1.jpg
│   │   │   ├── img_2.jpg
│   │   │   └── img_3.jpg
│   │   └── drone_video.mp4
│   └── ...
└── annotations/
  └── annotations.json

**samples/** — contains drone videos + object reference images  
**object_images/** — three RGB images from ground-level  
**drone_video.mp4** — 3–5 minute aerial footage (25 fps)  
**annotations/** — JSON file with labels  

---

## 2. Ground-Truth Annotation Format

Each record in `annotations.json` specifies when and where the target appears.

```json
{
  "video_id": "drone_video_01",
  "annotations": [
    {
      "bboxes": [
        {"frame": 370, "x1": 422, "y1": 310, "x2": 470, "y2": 355},
        {"frame": 371, "x1": 424, "y1": 312, "x2": 468, "y2": 354},
        {"frame": 372, "x1": 426, "y1": 314, "x2": 469, "y2": 356}
      ]
    }
  ]
}
```

Field Description:
- video_id: unique identifier of the drone video.
- bboxes: list of bounding boxes (x1, y1, x2, y2) per frame (absolute pixel coordinates).
- Each video may contain one or more visible intervals.

## 3. Expected Submission Format


- Predictions must follow the same schema as the ground-truth annotations.
- Every provided video must appear in the submission file — even if the object is not detected ("detections": []).

```json
[
   {
      "video_id": "drone_vid001",
      "detections": [
           {
               "bboxes": [
                   {"frame": 370, "x1": 422, "y1": 310, "x2": 470, "y2": 355},
                   {"frame": 371, "x1": 424, "y1": 312, "x2": 468, "y2": 354}
                ]
            }
       ]
   },
   {
       "video_id": "drone_video_002",
       "detections": []
   }
]
```