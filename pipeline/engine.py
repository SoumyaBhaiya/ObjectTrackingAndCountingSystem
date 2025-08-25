# pipeline/engine.py
from __future__ import annotations
import cv2
from typing import Optional, Dict, List
from core.base import Detector, Tracker, PostProcessor, Visualizer, Detection, Track, SimpleVisualizer

class VideoReader:
    def __init__(self, src: str | int):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {src}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self):
        ok, frame = self.cap.read()
        return ok, frame

    def release(self):
        self.cap.release()

class VideoWriter:
    def __init__(self, path: str, w: int, h: int, fps: float):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(path, fourcc, fps, (w, h))

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()

class Pipeline:
    def __init__(
        self,
        detector: Detector,
        tracker: Tracker,
        post: Optional[PostProcessor] = None,
        visualizer: Optional[Visualizer] = None,
        display: bool = True,
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
    ):
        self.detector = detector
        self.tracker = tracker
        self.post = post
        self.visualizer = visualizer or SimpleVisualizer()
        self.display = display
        self.output_path = output_path
        self.max_frames = max_frames

    def run(self, src: str | int = 0) -> Dict[str, float]:
        reader = VideoReader(src)
        writer = VideoWriter(self.output_path, reader.w, reader.h, reader.fps) if self.output_path else None

        self.detector.warmup()
        metrics: Dict[str, float] = {}
        frame_idx = 0

        try:
            while True:
                ok, frame = reader.read()
                if not ok:
                    break
                detections: List[Detection] = self.detector.detect(frame)
                tracks: List[Track] = self.tracker.update(detections)

                if self.post:
                    m_update = self.post.process(frame_idx, tracks)
                    metrics.update(m_update)

                # draw post overlays first (e.g., line)
                if self.post:
                    self.post.draw(frame)

                annotated = self.visualizer.annotate(frame, detections, tracks, metrics)

                if self.display:
                    cv2.imshow("RoadWatch", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                if writer:
                    writer.write(annotated)

                frame_idx += 1
                if self.max_frames is not None and frame_idx >= self.max_frames:
                    break
        finally:
            reader.release()
            if writer:
                writer.release()
            if self.display:
                cv2.destroyAllWindows()
        return metrics
