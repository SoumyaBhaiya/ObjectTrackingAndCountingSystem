# trackers/centroid_tracker.py
from __future__ import annotations
import numpy as np
from typing import List, Dict
from core.base import Tracker, Detection, Track, BBox

class CentroidTracker(Tracker):
    """
    Simple tracker: nearest-neighbor association on centroids + max distance gating.
    Good enough to start; later swap for DeepSORT/ByteTrack via the interface.
    """
    def __init__(self, max_dist: float = 50.0, max_age: int = 30):
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}
        self.max_dist = max_dist
        self.max_age = max_age

    def reset(self) -> None:
        self.next_id = 1
        self.tracks.clear()

    def _dist(self, c1, c2) -> float:
        return float(np.linalg.norm(np.array(c1) - np.array(c2)))

    def update(self, detections: List[Detection]) -> List[Track]:
        # Age existing tracks
        for t in self.tracks.values():
            t.age += 1

        det_centers = [d.bbox.center() for d in detections]
        det_used = set()
        # Associate by greedy nearest neighbor
        for tid, tr in list(self.tracks.items()):
            best_j = None
            best_d = 1e9
            for j, c in enumerate(det_centers):
                if j in det_used:
                    continue
                d = self._dist(tr.bbox.center(), c)
                if d < best_d:
                    best_d, best_j = d, j
            if best_j is not None and best_d <= self.max_dist:
                # Update track with detection
                det = detections[best_j]
                self.tracks[tid] = Track(track_id=tid, bbox=det.bbox, age=0, hits=tr.hits + 1, label=det.label)
                det_used.add(best_j)

        # Create new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j in det_used:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = Track(track_id=tid, bbox=det.bbox, age=0, hits=1, label=det.label)

        # Remove stale tracks
        for tid in list(self.tracks.keys()):
            if self.tracks[tid].age > self.max_age:
                del self.tracks[tid]

        return list(self.tracks.values())
