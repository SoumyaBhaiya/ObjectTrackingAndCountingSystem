# Object Tracking and Counting System

A simple modular **Python** computer vision pipeline for **real-time object detection (motion), tracking, and counting**. The system uses motion-based detection and centroid tracking to assign unique IDs to moving objects and count them as they cross a defined line.

---

## Features
- Motion-based detection using OpenCV background subtraction
- Centroid tracking to assign persistent IDs to moving objects
- Line-crossing counter to track objects crossing a virtual line
- Modular design with detector, tracker, post-processor, visualizer, engine. all made from scratch.
- Live annotated video output with tracking and metrics overlay
- Easy to extend for more advanced detectors or trackers in the future

---
