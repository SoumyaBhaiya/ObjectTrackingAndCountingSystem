#detectors/motion_detector.py
from __future__ import annotations
import cv2
import numpy as np
from typing import List
from core.base import Detector, Detection, BBox

class MotionDetector(Detector):
    """
    Lightweight starter detector using background subtraction.
    Replace later with a learned detector (e.g., YOLO) without changing pipeline.
    """
    def __init__(self, min_area: int = 500, dilate_iter: int = 2):
        self.backsub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)
        self.min_area = min_area
        self.dilate_iter = dilate_iter

    def warmup(self) -> None:
        #Nothing special.
        pass

    def detect(self, frame: np.ndarray) -> List[Detection]:
        fg = self.backsub.apply(frame)
        fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)[1]
        fg = cv2.dilate(fg, None, iterations=self.dilate_iter)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: List[Detection] = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h < self.min_area:
                continue
            detections.append(Detection(bbox=BBox(x, y, w, h), score=1.0, label="moving"))
        return detections
