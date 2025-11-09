from sort import Sort
import numpy as np

class Tracker:
    def __init__(self):
        self.tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

    def update(self, detections, frame_shape):
        # detections: [x1, y1, x2, y2, conf]
        if len(detections) == 0:
            dets = np.empty((0, 5))
        else:
            dets = detections
        tracked_objects = self.tracker.update(dets)
        results = []
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj
            results.append({
                "id": int(track_id),
                "bbox": [int(x1), int(y1), int(x2-x1), int(y2-y1)]
            })
        return results