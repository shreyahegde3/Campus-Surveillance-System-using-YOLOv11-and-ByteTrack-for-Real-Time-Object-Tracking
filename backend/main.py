from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
import supervision as sv
import asyncio
import json
import os
import logging
import sys
from pathlib import Path
import time
from collections import deque
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create an empty detections object
def create_empty_detections():
    return sv.Detections(
        xyxy=np.empty((0, 4)),
        confidence=np.array([]),
        class_id=np.array([]),
        tracker_id=None
    )

# Filter detections based on class names
def filter_detections(detections, model, include_keywords=None, exclude_keywords=None):
    if len(detections.xyxy) == 0:
        return create_empty_detections()
        
    mask = np.ones(len(detections.xyxy), dtype=bool)
    
    if include_keywords:
        include_mask = np.zeros(len(detections.xyxy), dtype=bool)
        for i, class_id in enumerate(detections.class_id):
            class_name = model.model.names[class_id].lower()
            if any(keyword in class_name for keyword in include_keywords):
                include_mask[i] = True
        mask = np.logical_and(mask, include_mask)
    
    if exclude_keywords:
        for i, class_id in enumerate(detections.class_id):
            class_name = model.model.names[class_id].lower()
            if any(keyword in class_name for keyword in exclude_keywords):
                mask[i] = False
    
    # Apply mask to all detection arrays
    return sv.Detections(
        xyxy=detections.xyxy[mask],
        confidence=detections.confidence[mask],
        class_id=detections.class_id[mask],
        tracker_id=detections.tracker_id[mask] if detections.tracker_id is not None else None
    )

# Check if YOLO model exists in current or parent directory
def find_model(model_name):
    current_dir = Path('.')
    possible_locations = [
        current_dir / model_name,
        current_dir.parent / model_name,
        Path(os.path.expanduser('~/OneDrive/Desktop/CAPSTONE')) / model_name,
        Path(os.path.expanduser('~/OneDrive/Desktop/CAPSTONE')) / "CODE" / model_name,
        Path(os.path.expanduser('~/OneDrive/Desktop/CAPSTONE/YOLO')) / model_name,
    ]
    
    for path in possible_locations:
        if path.exists():
            logger.info(f"Found model at {path}")
            return str(path.absolute())
    
    logger.warning(f"Could not find {model_name} in expected locations")
    return model_name

# Class to track and stabilize detections over time
class DetectionTracker:
    def __init__(self, history_size=5, persistence_time=1.0):
        self.history = deque(maxlen=history_size)
        self.last_detection_time = {}  # Track when we last saw each class
        self.persistence_time = persistence_time    # How long to keep showing a detection after it disappears (seconds)
        self.model = None  # Store reference to the model
        
    def update(self, detections, model, current_time):
        # Store model reference
        self.model = model
        
        # Add current detections to history
        if len(detections.xyxy) > 0:
            self.history.append((detections, current_time))
            
            # Update last seen time for each class
            for class_id in detections.class_id:
                class_name = model.model.names[class_id]
                self.last_detection_time[class_name] = current_time
                
        # Return stabilized detections
        return self.get_stable_detections(current_time)
    
    def get_stable_detections(self, current_time):
        if not self.history or self.model is None:
            return create_empty_detections()
            
        # Start with most recent detections
        recent_detections, _ = self.history[-1]
        
        all_boxes = []
        all_confidences = []
        all_classes = []
        all_labels = []
        
        # Include persistent detections that haven't been seen recently but are still within persistence time
        for class_name, last_time in list(self.last_detection_time.items()):
            if current_time - last_time <= self.persistence_time:
                # Find most recent detection of this class
                for hist_det, hist_time in reversed(self.history):
                    for i, class_id in enumerate(hist_det.class_id):
                        if class_name == self.model.model.names[class_id]:
                            # Only add if we haven't already added this class
                            if class_name not in all_labels:
                                all_boxes.append(hist_det.xyxy[i])
                                all_confidences.append(hist_det.confidence[i])
                                all_classes.append(hist_det.class_id[i])
                                all_labels.append(class_name)
                            break
            else:
                # Remove expired detections
                del self.last_detection_time[class_name]
        
        if not all_boxes:
            return create_empty_detections()
            
        return sv.Detections(
            xyxy=np.array(all_boxes),
            confidence=np.array(all_confidences),
            class_id=np.array(all_classes),
            tracker_id=None
        )

# Background inference worker for non-blocking model execution
class ModelInferenceWorker:
    def __init__(self, model, input_queue, output_queue, name="worker"):
        self.model = model
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.name = name
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, name=name)
        self.thread.daemon = True
        
    def start(self):
        self.thread.start()
        
    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
    
    def _worker_loop(self):
        while self.running:
            try:
                # Get task with 0.1s timeout to allow checking if we should stop
                task = self.input_queue.get(timeout=0.1)
                if task is None:
                    continue
                
                frame_id, frame = task
                
                # Run model inference
                start_time = time.time()
                results = self.model(frame, verbose=False)[0]
                inference_time = time.time() - start_time
                
                # Convert to detections
                detections = sv.Detections.from_yolov8(results)
                
                # Put result in output queue
                self.output_queue.put((frame_id, detections, inference_time))
                
                # Mark task as done
                self.input_queue.task_done()
                
            except queue.Empty:
                # Just continue if no task
                pass
            except Exception as e:
                logger.error(f"Error in {self.name} worker: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    # First try importing ultralytics with version check
    import pkg_resources
    ultralytics_version = pkg_resources.get_distribution("ultralytics").version
    logger.info(f"Detected Ultralytics version: {ultralytics_version}")
    
    from ultralytics import YOLO
    
    # Locate the models
    primary_model_path = find_model("yolov8n-oiv7.pt")
    bag_model_path = find_model("yolov11n.pt")  # Use standard YOLOv8 model for bags
    
    # Try to load the primary model
    logger.info(f"Loading primary model from {primary_model_path}...")
    primary_model = YOLO(primary_model_path)
    
    # Try to load the bag detection model
    try:
        logger.info(f"Loading bag detection model from {bag_model_path}...")
        bag_model = YOLO(bag_model_path)
        use_dual_model = True
        logger.info("Successfully loaded both models!")
    except Exception as e:
        logger.warning(f"Failed to load bag model: {str(e)}")
        use_dual_model = False
        bag_model = None
        
except Exception as e:
    logger.error(f"Error initializing YOLO models: {str(e)}")
    sys.exit(1)

# Initialize annotators
box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=2,
    text_scale=1
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        # Performance optimization: Configure OpenCV for optimal speed
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # DirectShow API for Windows
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Reduced resolution
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Reduced resolution
        cap.set(cv2.CAP_PROP_FPS, 30)            # Target FPS
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # Minimize buffer size for lower latency
        
        # Start up background worker threads for model inference
        primary_input_queue = queue.Queue(maxsize=1)  # Only queue one frame at a time
        primary_output_queue = queue.Queue()
        primary_worker = ModelInferenceWorker(primary_model, primary_input_queue, primary_output_queue, "primary_model")
        primary_worker.start()
        
        if use_dual_model:
            bag_input_queue = queue.Queue(maxsize=1)  # Only queue one frame at a time
            bag_output_queue = queue.Queue()
            bag_worker = ModelInferenceWorker(bag_model, bag_input_queue, bag_output_queue, "bag_model")
            bag_worker.start()
        
        # Initialize detection trackers
        primary_tracker = DetectionTracker(history_size=5)
        bag_tracker = DetectionTracker(history_size=5)
        
        # Processing state
        frame_id = 0
        last_primary_id = -1
        last_bag_id = -1
        
        # Cache for detection results
        cached_primary_detections = create_empty_detections()
        cached_bag_detections = create_empty_detections()
        
        # Resize dimensions for processing
        process_width = 384
        process_height = 288
        
        # Last frame optimization
        last_annotated_frame = None
        last_labels = []
        
        # Fps calculation
        frame_times = deque(maxlen=30)
        last_frame_time = time.time()
        
        # Define keywords for bag detection - moved outside the conditional block
        bag_keywords = ['bag', 'handbag', 'backpack', 'luggage']
        
        while True:
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.01)
                continue
            
            # Calculate FPS
            current_time = time.time()
            frame_times.append(current_time - last_frame_time)
            last_frame_time = current_time
            fps = len(frame_times) / sum(frame_times) if frame_times else 0
            
            # Increment frame counter
            frame_id += 1
            
            # Resize frame for inference
            small_frame = cv2.resize(frame, (process_width, process_height))
            
            # Submit frame for processing if not already waiting
            if primary_input_queue.qsize() == 0:
                try:
                    # Non-blocking put
                    primary_input_queue.put_nowait((frame_id, small_frame))
                except queue.Full:
                    pass  # Skip if queue is full
            
            # Check for primary model results
            primary_detections = None
            while not primary_output_queue.empty():
                result_id, detections, inference_time = primary_output_queue.get()
                logger.debug(f"Primary model: {inference_time*1000:.1f}ms, FPS: {1/inference_time:.1f}")
                
                # Scale detection coordinates to original frame size
                if len(detections.xyxy) > 0:
                    scale_x = frame.shape[1] / process_width
                    scale_y = frame.shape[0] / process_height
                    detections.xyxy[:, [0, 2]] *= scale_x
                    detections.xyxy[:, [1, 3]] *= scale_y
                
                last_primary_id = result_id
                primary_detections = detections
                
                # Update tracker
                cached_primary_detections = primary_tracker.update(detections, primary_model, current_time)
            
            # If we didn't get new primary detections, use cached ones
            if primary_detections is None:
                primary_detections = cached_primary_detections
            
            # Process bag detection in dual model mode
            if use_dual_model:
                # Submit frame for bag model if not already waiting
                if bag_input_queue.qsize() == 0:
                    # Only run bag model every 3rd frame to reduce load
                    if frame_id % 3 == 0:
                        try:
                            # Non-blocking put
                            bag_input_queue.put_nowait((frame_id, small_frame))
                        except queue.Full:
                            pass  # Skip if queue is full
                
                # Check for bag model results
                bag_detections = None
                while not bag_output_queue.empty():
                    result_id, detections, inference_time = bag_output_queue.get()
                    logger.debug(f"Bag model: {inference_time*1000:.1f}ms, FPS: {1/inference_time:.1f}")
                    
                    # Scale detection coordinates to original frame size
                    if len(detections.xyxy) > 0:
                        scale_x = frame.shape[1] / process_width
                        scale_y = frame.shape[0] / process_height
                        detections.xyxy[:, [0, 2]] *= scale_x
                        detections.xyxy[:, [1, 3]] *= scale_y
                    
                    last_bag_id = result_id
                    bag_detections = detections
                    
                    # Filter bag detections
                    filtered_bag_detections = filter_detections(
                        detections, 
                        bag_model, 
                        include_keywords=bag_keywords
                    )
                    
                    # Apply confidence threshold
                    if len(filtered_bag_detections.xyxy) > 0:
                        confidence_mask = filtered_bag_detections.confidence > 0.3
                        filtered_bag_detections = sv.Detections(
                            xyxy=filtered_bag_detections.xyxy[confidence_mask] if len(filtered_bag_detections.xyxy) > 0 and any(confidence_mask) else np.empty((0, 4)),
                            confidence=filtered_bag_detections.confidence[confidence_mask] if len(filtered_bag_detections.confidence) > 0 and any(confidence_mask) else np.array([]),
                            class_id=filtered_bag_detections.class_id[confidence_mask] if len(filtered_bag_detections.class_id) > 0 and any(confidence_mask) else np.array([]),
                            tracker_id=None
                        )
                    
                    # Update cached bag detections through tracker for stability
                    cached_bag_detections = bag_tracker.update(filtered_bag_detections, bag_model, current_time)
                
                # If we didn't get new bag detections, use cached ones
                if bag_detections is None:
                    filtered_bag_detections = cached_bag_detections
                
                # Filter primary detections to exclude bags
                filtered_primary = filter_detections(
                    primary_detections, 
                    primary_model, 
                    exclude_keywords=bag_keywords
                )
                
                # Create labels for non-bag detections
                labels = []
                
                # Add labels for primary detections (non-bag)
                if len(filtered_primary.xyxy) > 0:
                    for _, confidence, class_id, _ in zip(
                        filtered_primary.xyxy, 
                        filtered_primary.confidence, 
                        filtered_primary.class_id,
                        filtered_primary.tracker_id if filtered_primary.tracker_id is not None else [None] * len(filtered_primary.xyxy)
                    ):
                        labels.append(f"{primary_model.model.names[class_id]} {confidence:0.2f}")
                
                # Add bag detection labels from specialized model
                if len(filtered_bag_detections.xyxy) > 0:
                    for _, confidence, class_id, _ in zip(
                        filtered_bag_detections.xyxy, 
                        filtered_bag_detections.confidence, 
                        filtered_bag_detections.class_id,
                        filtered_bag_detections.tracker_id if filtered_bag_detections.tracker_id is not None else [None] * len(filtered_bag_detections.xyxy)
                    ):
                        # Highlight bag detections from specialized model
                        labels.append(f"*{bag_model.model.names[class_id]}* {confidence:0.2f}")
                
                # Combine detections
                combined_detections = create_empty_detections()
                
                # Only combine if we have detections to combine
                has_primary = len(filtered_primary.xyxy) > 0
                has_bags = len(filtered_bag_detections.xyxy) > 0
                
                if has_primary or has_bags:
                    # Combine xyxy boxes
                    if has_primary and has_bags:
                        combined_xyxy = np.vstack([filtered_primary.xyxy, filtered_bag_detections.xyxy])
                        combined_confidence = np.hstack([filtered_primary.confidence, filtered_bag_detections.confidence])
                        combined_class_id = np.hstack([filtered_primary.class_id, filtered_bag_detections.class_id])
                    elif has_primary:
                        combined_xyxy = filtered_primary.xyxy
                        combined_confidence = filtered_primary.confidence
                        combined_class_id = filtered_primary.class_id
                    else:  # has_bags
                        combined_xyxy = filtered_bag_detections.xyxy
                        combined_confidence = filtered_bag_detections.confidence
                        combined_class_id = filtered_bag_detections.class_id
                    
                    # Create combined detections
                    combined_detections = sv.Detections(
                        xyxy=combined_xyxy,
                        confidence=combined_confidence,
                        class_id=combined_class_id,
                        tracker_id=None
                    )
            else:
                # If not using dual model, just use all primary detections
                combined_detections = primary_detections
                
                # Create labels for all detections
                labels = []
                if len(primary_detections.xyxy) > 0:
                    for _, confidence, class_id, _ in zip(
                        primary_detections.xyxy, 
                        primary_detections.confidence, 
                        primary_detections.class_id,
                        primary_detections.tracker_id if primary_detections.tracker_id is not None else [None] * len(primary_detections.xyxy)
                    ):
                        labels.append(f"{primary_model.model.names[class_id]} {confidence:0.2f}")
            
            # Cache labels
            last_labels = labels
            
            # Annotate frame with combined detections
            annotated_frame = frame.copy()
            annotated_frame = box_annotator.annotate(
                scene=annotated_frame,
                detections=combined_detections,
                labels=labels
            )
            
            # Add FPS counter to the frame
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Cache the annotated frame
            last_annotated_frame = annotated_frame
            
            # Convert frame to base64 with reduced quality for faster transmission
            _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Send frame and detections
            await websocket.send_json({
                "frame": frame_base64,
                "detections": labels
            })
            
            # Minimal sleep to allow other tasks
            await asyncio.sleep(0.001)
            
    except Exception as e:
        logger.error(f"Error in websocket endpoint: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # Clean up resources
        if 'primary_worker' in locals():
            primary_worker.stop()
        if use_dual_model and 'bag_worker' in locals():
            bag_worker.stop()
        if 'cap' in locals() and cap is not None:
            cap.release()
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)