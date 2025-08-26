# app.py
import argparse
from detectors.motion_detector import MotionDetector
from trackers.centroid_tracker import CentroidTracker
from post.line_counter import LineCounter
from pipeline.engine import Pipeline

def parse_args(): #we can add and change these when running the app from command prompt.
    p = argparse.ArgumentParser(description="RoadWatch – Traffic Analytics")
    p.add_argument("--src", default=0, help="Video source (path or camera index)")
    p.add_argument("--output", default=None, help="Optional path to save annotated video (e.g., out.mp4)")
    p.add_argument("--max-frames", type=int, default=None, help="Limit frames for quick tests")
    p.add_argument("--line", type=int, nargs=4, metavar=("x1","y1","x2","y2"),
                   default=[400, 100, 400, 600], help="Counting line coordinates")
    p.add_argument("--min-area", type=int, default=800, help="Min area for motion detector")
    p.add_argument("--dilate", type=int, default=2, help="Dilate iterations for motion mask")
    p.add_argument("--no-display", action="store_true", help="Disable window display")
    return p.parse_args()

def main():
    args = parse_args()

    detector = MotionDetector(min_area=args.min_area, dilate_iter=args.dilate)
    tracker = CentroidTracker(max_dist=60.0, max_age=25)
    post = LineCounter((args.line[0], args.line[1]), (args.line[2], args.line[3]))

    pipeline = Pipeline(
        detector=detector,
        tracker=tracker,
        post=post,
        display=not args.no_display,
        output_path=args.output,
        max_frames=args.max_frames
    )
    metrics = pipeline.run(src=int(args.src) if str(args.src).isdigit() else args.src)
    print("Final metrics:", metrics)

if __name__ == "__main__":
    main()
