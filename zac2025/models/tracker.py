import numpy as np
from collections import deque


class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        from scipy.optimize import linear_sum_assignment
        from scipy.spatial.distance import cdist

        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.bbox = bbox
        self.history = deque(maxlen=30)
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.time_since_update = 0
        self.confidence = 1.0

        self.kf_x = (bbox[0] + bbox[2]) / 2
        self.kf_y = (bbox[1] + bbox[3]) / 2
        self.kf_w = bbox[2] - bbox[0]
        self.kf_h = bbox[3] - bbox[1]
        self.kf_vx = 0
        self.kf_vy = 0

    def update(self, bbox, confidence=1.0):
        self.time_since_update = 0
        self.history.append(bbox)
        self.hits += 1
        self.hit_streak += 1
        self.bbox = bbox
        self.confidence = confidence

        new_x = (bbox[0] + bbox[2]) / 2
        new_y = (bbox[1] + bbox[3]) / 2

        self.kf_vx = new_x - self.kf_x
        self.kf_vy = new_y - self.kf_y

        self.kf_x = new_x
        self.kf_y = new_y
        self.kf_w = bbox[2] - bbox[0]
        self.kf_h = bbox[3] - bbox[1]

    def predict(self):
        self.age += 1
        self.time_since_update += 1
        self.hit_streak = 0

        self.kf_x += self.kf_vx
        self.kf_y += self.kf_vy

        x1 = self.kf_x - self.kf_w / 2
        y1 = self.kf_y - self.kf_h / 2
        x2 = self.kf_x + self.kf_w / 2
        y2 = self.kf_y + self.kf_h / 2

        self.bbox = [x1, y1, x2, y2]
        return self.bbox

    def get_state(self):
        return self.bbox


class ByteTracker:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, detections):
        self.frame_count += 1

        for trk in self.trackers:
            trk.predict()

        if len(detections) > 0:
            matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(
                detections, self.trackers
            )

            for m in matched:
                det_idx, trk_idx = m
                det = detections[det_idx]
                self.trackers[trk_idx].update(det['bbox'], det.get('confidence', 1.0))

            for i in unmatched_dets:
                trk = KalmanBoxTracker(detections[i]['bbox'])
                self.trackers.append(trk)

        ret = []
        for trk in self.trackers:
            if trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits:
                ret.append({
                    'bbox': trk.get_state(),
                    'id': trk.id,
                    'confidence': trk.confidence
                })

        self.trackers = [trk for trk in self.trackers if trk.time_since_update < self.max_age]

        return ret

    def _iou(self, bbox1, bbox2):
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)

        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        union_area = bbox1_area + bbox2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def _associate_detections_to_trackers(self, detections, trackers):
        if len(trackers) == 0:
            return [], list(range(len(detections))), []

        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = self._iou(det['bbox'], trk.get_state())

        from scipy.optimize import linear_sum_assignment

        matched_indices = []
        if min(iou_matrix.shape) > 0:
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)

            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_threshold:
                    matched_indices.append([r, c])

        unmatched_detections = [d for d in range(len(detections))
                                if d not in [m[0] for m in matched_indices]]
        unmatched_trackers = [t for t in range(len(trackers))
                              if t not in [m[1] for m in matched_indices]]

        return matched_indices, unmatched_detections, unmatched_trackers

    def get_best_track(self):
        if len(self.trackers) == 0:
            return None

        best_tracker = max(self.trackers, key=lambda t: t.confidence * t.hit_streak)

        if best_tracker.hit_streak >= self.min_hits:
            return {
                'bbox': best_tracker.get_state(),
                'id': best_tracker.id,
                'confidence': best_tracker.confidence
            }

        return None


class SimpleTracker:
    def __init__(self, iou_threshold=0.3):
        self.iou_threshold = iou_threshold
        self.prev_bbox = None
        self.confidence = 0.0
        self.lost_frames = 0

    def update(self, bbox, confidence):
        self.prev_bbox = bbox
        self.confidence = confidence
        self.lost_frames = 0

    def predict(self):
        if self.prev_bbox is None:
            return None

        self.lost_frames += 1

        if self.lost_frames > 20:
            self.prev_bbox = None
            self.confidence = 0.0
            return None

        decay = max(0.5, 1.0 - (self.lost_frames * 0.05))
        self.confidence *= decay

        return {
            'bbox': self.prev_bbox,
            'confidence': self.confidence
        }

    def reset(self):
        self.prev_bbox = None
        self.confidence = 0.0
        self.lost_frames = 0
