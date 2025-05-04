import cv2
import argparse
import os
import threading
import queue
from pathlib import Path

from ultralytics import YOLO
import supervision as sv
import numpy as np


ZONE_POLYGON = np.array([
    [0, 0],
    [0.5, 0],
    [0.5, 1],
    [0, 1]
])


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution", 
        default=[1280, 720], 
        nargs=2, 
        type=int
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        type=str,
        help="Directory to save output frames"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable GUI display (save frames instead)"
    )
    parser.add_argument(
        "--save-frequency",
        default=5,
        type=int,
        help="Save every Nth frame (higher values reduce disk I/O)"
    )
    parser.add_argument(
        "--conf-threshold",
        default=0.3,
        type=float,
        help="Confidence threshold for detections"
    )
    args = parser.parse_args()
    return args


def capture_frames(cap, frame_queue, stop_event):
    """Thread function to capture frames from webcam"""
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Failed to get frame from camera")
            break
        
        # If queue is full, remove oldest frame
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        
        frame_queue.put(frame)


def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    
    # Set additional camera properties to reduce lag
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size
    cap.set(cv2.CAP_PROP_FPS, 30)  # Request 30 FPS if camera supports it

    # Create frame queue and thread for async frame capture
    frame_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    capture_thread = threading.Thread(
        target=capture_frames, 
        args=(cap, frame_queue, stop_event),
        daemon=True
    )
    capture_thread.start()

    # Load YOLO model
    model = YOLO("yolov8n-oiv7.pt")
    
    # Set up annotators
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
    zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=tuple(args.webcam_resolution))
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone, 
        color=sv.Color.red(),
        thickness=2,
        text_thickness=4,
        text_scale=2
    )

    frame_count = 0
    try:
        while True:
            try:
                frame = frame_queue.get(timeout=1.0)
            except queue.Empty:
                print("No frames available, waiting...")
                continue

            # Run YOLO inference with confidence threshold
            results = model(frame, conf=args.conf_threshold, agnostic_nms=True)[0]
            detections = sv.Detections.from_yolov8(results)
            
            # Only process frames for display/save
            labels = [
                f"{model.model.names[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, _
                in detections
            ]
            
            # Only annotate if we have detections or need to display
            if not args.no_display or len(detections) > 0 or frame_count % args.save_frequency == 0:
                annotated_frame = frame.copy()
                annotated_frame = box_annotator.annotate(
                    scene=annotated_frame, 
                    detections=detections, 
                    labels=labels
                )

                zone.trigger(detections=detections)
                annotated_frame = zone_annotator.annotate(scene=annotated_frame)

                # Save frame selectively based on save_frequency
                if frame_count % args.save_frequency == 0:
                    output_path = os.path.join(args.output_dir, f"frame_{frame_count:04d}.jpg")
                    cv2.imwrite(output_path, annotated_frame)
                    print(f"Saved frame to {output_path}")
                
                if len(labels) > 0:
                    print(f"Detected objects: {labels}")
                
                # Display if not disabled
                if not args.no_display:
                    try:
                        cv2.imshow("PES University Campus Surveillance", annotated_frame)
                        key = cv2.waitKey(1)  # 1ms wait instead of 30ms
                        if key == 27:  # ESC key
                            break
                    except cv2.error:
                        print("Warning: Could not display frame with cv2.imshow(). "
                            "Running in headless mode and saving frames instead.")
                        args.no_display = True  # Switch to headless mode if display fails
            
            frame_count += 1

    except KeyboardInterrupt:
        print("Stopping detection...")
    finally:
        stop_event.set()  # Signal capture thread to stop
        capture_thread.join(timeout=1.0)  # Wait for capture thread to finish
        cap.release()
        cv2.destroyAllWindows()
        print(f"Processed {frame_count} frames. Output saved to {args.output_dir}/")


if __name__ == "__main__":
    main()