# core/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np

# ---------- Data models ----------
@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int
    def to_xyxy(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)
    def center(self) -> Tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

@dataclass
class Detection:
    bbox: BBox
    score: float
    label: str = "object"

@dataclass
class Track:
    track_id: int
    bbox: BBox
    age: int = 0
    hits: int = 0
    label: str = "object"

# ---------- Interfaces ------
class Detector(ABC):
    @abstractmethod
    def warmup(self) -> None: ...
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]: ...

class Tracker(ABC):
    @abstractmethod
    def update(self, detections: List[Detection]) -> List[Track]: ...
    @abstractmethod
    def reset(self) -> None: ...

class PostProcessor(ABC):
    @abstractmethod
    def process(self, frame_idx: int, tracks: List[Track]) -> Dict[str, float]: ...
    @abstractmethod
    def draw(self, frame: np.ndarray) -> None: ...

class Visualizer(ABC):
    @abstractmethod
    def annotate(self, frame: np.ndarray, detections: List[Detection], tracks: List[Track], metrics: Dict[str, float]) -> np.ndarray: ...

# ---------- Simple visualizer ----------
class SimpleVisualizer(Visualizer):
    def __init__(self, show_ids: bool = True) -> None:
        self.show_ids = show_ids

    def annotate(self, frame, detections, tracks, metrics):
        import cv2
        out = frame.copy()
        # Detections (thin boxes)
        for det in detections:
            x1, y1, x2, y2 = det.bbox.to_xyxy()
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 1)
        # Tracks (thick boxes)
        for tr in tracks:
            x1, y1, x2, y2 = tr.bbox.to_xyxy()
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 2)
            if self.show_ids:
                cv2.putText(out, f"ID {tr.track_id}", (x1, max(10, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        # HUD
        y = 20
        for k, v in metrics.items():
            cv2.putText(out, f"{k}: {v:.0f}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(out, f"{k}: {v:.0f}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            y += 22
        return out
