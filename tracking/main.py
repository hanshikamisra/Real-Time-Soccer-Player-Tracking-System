import os
import sys
import time
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import our custom modules
from track_new import SoccerTracker
from visualizer import visualize_frame
from speed_estimator import calculate_speeds_from_tracks

# --- PATH SETUP ---
ROOT_DIR = os.getcwd()
YOLO_DIR = os.path.join(ROOT_DIR, "yolov5")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

sys.path.insert(0, YOLO_DIR)

# --- CONFIG ---
VIDEO_PATH = r"C:\Users\hansh\Desktop\Soccer_tracking\data\test.mp4"
MODEL_PATH = os.path.join(YOLO_DIR, "yolov5s.pt")  # Can be custom model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_CSV_PATH = os.path.join(OUTPUT_DIR, "tracking_data.csv")
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5
MAX_AGE = 30  # Frames to keep a track alive without detection
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, "tracked_output.mp4")

# Field dimensions in meters (standard soccer field)
FIELD_WIDTH_M = 105
FIELD_HEIGHT_M = 68

def main():

    print("Starting Soccer Tracking System...")
    print(f"Looking for video at: {VIDEO_PATH}")
    print(f"Using model at: {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    
    # Check if video file exists
    if not os.path.exists(VIDEO_PATH):
        print(f"ERROR: Video file not found at {VIDEO_PATH}")
        return
    
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        return
    
    # Continue with the rest of your code...
    # Create soccer tracker
    tracker = SoccerTracker(
        model_path=MODEL_PATH, 
        device=DEVICE, 
        max_age=MAX_AGE,
        conf_threshold=CONF_THRESHOLD,
        iou_threshold=IOU_THRESHOLD
    )
    
    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
      print(f"ERROR: Could not open video file at {VIDEO_PATH}")
      sys.exit(1)
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))
    
    # Store tracking data
    tracking_data = []
    
    frame_id = 0
    prev_time = 0
    fps_values = []
    
    # Use tqdm for progress bar
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate FPS
            current_time = time.time()
            if prev_time > 0:
                fps_values.append(1.0 / (current_time - prev_time))
            prev_time = current_time
            current_fps = sum(fps_values[-10:]) / min(len(fps_values), 10) if fps_values else 0
            
            # Run detector and tracker
            tracks, detections = tracker.detect_and_track(frame)
            
            # Process tracks and store data
            for track in tracks:
                if not track.is_confirmed():
                    continue
                    
                track_id = track.track_id
                track_class = track.get_det_class() if hasattr(track, 'get_det_class') else None
                
                # Get bounding box
                ltrb = track.to_ltrb()
                l, t, r, b = map(int, ltrb)
                
                # Calculate center point
                center_x = (l + r) / 2
                center_y = (t + b) / 2
                
                # Add to tracking data
                tracking_data.append({
                    "frame": frame_id,
                    "track_id": track_id,
                    "class": track_class,
                    "x": center_x,
                    "y": center_y,
                    "x1": l,
                    "y1": t,
                    "x2": r,
                    "y2": b,
                    "width": r - l,
                    "height": b - t
                })
            
            # Visualize results
            vis_frame = visualize_frame(frame, tracks, frame_id, current_fps)
            
            # Write frame to output video
            out.write(vis_frame)
            
            # Display frame (optional)
            cv2.imshow("Soccer Tracking", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            
            frame_id += 1
            pbar.update(1)
    
    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Save tracking data
    df = pd.DataFrame(tracking_data)
    
    # Calculate speeds and add to dataframe
    df = calculate_speeds_from_tracks(df, fps, FIELD_WIDTH_M, FIELD_HEIGHT_M, width, height)
    
    # Save the final dataframe
    df.to_csv(SAVE_CSV_PATH, index=False)
    print(f"Tracking data saved to {SAVE_CSV_PATH}")
    
    print(f"Processing complete! Average FPS: {sum(fps_values) / len(fps_values):.2f}")
    print(f"Output video saved to {OUTPUT_VIDEO_PATH}")
    print(f"Run heatmap.py and voronoi.py to generate visualizations")


if __name__ == "__main__":
    main()