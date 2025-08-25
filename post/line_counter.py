# post/line_counter.py
from __future__ import annotations
from typing import Dict, List, Tuple
from core.base import PostProcessor, Track

def _crossed(a: Tuple[int, int], b: Tuple[int, int], p: Tuple[int, int]) -> bool:
    # Determine if point p is above/left vs. below/right of oriented segment a->b using cross product sign
    (x1, y1), (x2, y2) = a, b
    (px, py) = p
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1) > 0

class LineCounter(PostProcessor):
    """
    Counts tracks when their centers cross an oriented line.
    """
    def __init__(self, a: Tuple[int, int], b: Tuple[int, int]):
        self.a, self.b = a, b
        self.side_by_id: Dict[int, bool] = {}
        self.count = 0

    def process(self, frame_idx: int, tracks: List[Track]) -> Dict[str, float]:
        for tr in tracks:
            now_side = _crossed(self.a, self.b, tr.bbox.center())
            if tr.track_id not in self.side_by_id:
                self.side_by_id[tr.track_id] = now_side
                continue
            prev_side = self.side_by_id[tr.track_id]
            if now_side != prev_side and tr.hits > 2:
                self.count += 1
            self.side_by_id[tr.track_id] = now_side
        return {"count": float(self.count)}

    def draw(self, frame) -> None:
        import cv2
        cv2.line(frame, self.a, self.b, (0, 255, 255), 2)
        cv2.circle(frame, self.a, 4, (0, 255, 255), -1)
        cv2.circle(frame, self.b, 4, (0, 255, 255), -1)
