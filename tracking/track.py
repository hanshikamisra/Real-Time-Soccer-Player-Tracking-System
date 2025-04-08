import os
import sys
import time
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- PATH SETUP ---
ROOT_DIR = os.getcwd()
YOLO_DIR = os.path.join(ROOT_DIR, "yolov5")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

sys.path.insert(0, YOLO_DIR)

# --- IMPORTS FROM YOLOv5 ---
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox

# --- IMPORT TRACKER ---
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- CONFIG ---
VIDEO_PATH = r"C:\Users\hansh\Desktop\Soccer_tracking\data\test.mp4"
MODEL_PATH = os.path.join(YOLO_DIR, "yolov5s.pt")  # Can be custom model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_CSV_PATH = os.path.join(OUTPUT_DIR, "tracking_data.csv")
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5
MAX_AGE = 30  # Frames to keep a track alive without detection
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, "tracked_output.mp4")

# Class mapping
CLASS_NAMES = {
    0: "player",  # person in standard YOLO
    1: "referee", # custom class in our model
    32: "ball"    # custom class in our model
}

# --- CUSTOM TRACKER CLASS ---
class SoccerTracker:
    def __init__(self, model_path, device, max_age=30):
        # Load YOLOv5 model
        print("Initializing YOLOv5 model...")
        self.model = DetectMultiBackend(model_path, device=device)
        self.model.eval()
        self.device = device

        # Initialize DeepSORT tracker
        self.tracker = DeepSort(max_age=max_age, 
                               n_init=3,
                               nms_max_overlap=1.0,
                               max_cosine_distance=0.3,
                               nn_budget=100)
        
        # Store tracking history
        self.tracking_data = []
        
        # FPS calculation
        self.prev_time = 0
        self.fps_history = []
        
    def detect_and_track(self, frame, frame_id):
        # Time tracking for performance analysis
        start_time = time.time()
        
        # --- Preprocess frame ---
        img_resized = letterbox(frame, new_shape=640)[0]
        img_rgb = img_resized[:, :, ::-1]  # BGR to RGB
        img_rgb = np.ascontiguousarray(img_rgb)  # Ensure positive strides

        img_tensor = torch.from_numpy(img_rgb).to(self.device).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # HWC -> CHW + batch

        # Original frame dimensions for scaling
        h, w = frame.shape[:2]
        self.img_width, self.img_height = w, h

        # --- Run YOLOv5 detection ---
        with torch.no_grad():
            preds = self.model(img_tensor)
            preds = non_max_suppression(preds, conf_thres=CONF_THRESHOLD, iou_thres=IOU_THRESHOLD)[0]

        # --- Prepare detections for tracker ---
        detections = []
        if preds is not None and len(preds):
            # Scale boxes to original image size
            preds[:, :4] = scale_boxes(img_tensor.shape[2:], preds[:, :4], frame.shape).round()

            for *xyxy, conf, cls in preds:
                cls = int(cls)
                # Filter only players, referees, and ball
                if cls not in [0, 1, 32]:  # Adjust based on your model's class indices
                    continue

                x1, y1, x2, y2 = map(int, xyxy)
                w, h = x2 - x1, y2 - y1
                
                # Skip tiny detections (likely noise)
                if w < 5 or h < 5:
                    continue

                bbox = [x1, y1, w, h]
                detections.append((bbox, float(conf), cls))

        # --- Run DeepSORT tracker ---
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        # Process each track
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            track_class = track.get_det_class() if hasattr(track, 'get_det_class') else None
            
            ltrb = track.to_ltrb()
            l, t, r, b = map(int, ltrb)
            
            # Calculate center point (for position tracking)
            center_x = (l + r) / 2
            center_y = (t + b) / 2
            
            # --- Record tracking data ---
            self.tracking_data.append({
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
        
        # Calculate FPS
        current_time = time.time()
        if self.prev_time > 0:
            current_fps = 1.0 / (current_time - self.prev_time)
            self.fps_history.append(current_fps)
        self.prev_time = current_time
        
        return tracks
    
    def save_data(self, csv_path):
        df = pd.DataFrame(self.tracking_data)
        df.to_csv(csv_path, index=False)
        print(f"Tracking data saved to {csv_path}")
        return df

    def get_average_fps(self):
        if not self.fps_history:
            return 0
        return sum(self.fps_history) / len(self.fps_history)


def visualize_frame(frame, tracks, frame_id, fps):
    """Draw bounding boxes, IDs, and speed information on the frame"""
    # Create a copy of the frame to draw on
    vis_frame = frame.copy()
    
    # Draw tracking information
    for track in tracks:
        if not track.is_confirmed():
            continue
            
        track_id = track.track_id
        track_class = track.get_det_class() if hasattr(track, 'get_det_class') else None
        
        # Get bounding box
        ltrb = track.to_ltrb()
        l, t, r, b = map(int, ltrb)
        
        # Set color based on class
        if track_class == 32:  # Ball
            color = (0, 0, 255)  # Red for ball
        elif track_class == 1:  # Referee
            color = (255, 255, 0)  # Yellow for referee
        else:  # Players
            color = (0, 255, 0)  # Green for players
        
        # Draw bounding box
        cv2.rectangle(vis_frame, (l, t), (r, b), color, 2)
        
        # Draw ID
        label = f"ID:{track_id}"
        if track_class is not None and track_class in CLASS_NAMES:
            label += f" {CLASS_NAMES[track_class]}"
        
        cv2.putText(vis_frame, label, (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw center point
        center_x, center_y = int((l + r) / 2), int((t + b) / 2)
        cv2.circle(vis_frame, (center_x, center_y), 4, (255, 0, 255), -1)
        
    # Draw FPS
    cv2.putText(vis_frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Draw frame number
    cv2.putText(vis_frame, f"Frame: {frame_id}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return vis_frame


def main():
    # Create soccer tracker
    soccer_tracker = SoccerTracker(MODEL_PATH, DEVICE, MAX_AGE)
    
    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    assert cap.isOpened(), "Could not open video file"
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))
    
    frame_id = 0
    
    # Use tqdm for progress bar
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detector and tracker
            tracks = soccer_tracker.detect_and_track(frame, frame_id)
            
            # Visualize results
            vis_frame = visualize_frame(frame, tracks, frame_id, soccer_tracker.get_average_fps())
            
            # Write frame to output video
            out.write(vis_frame)
            
            # Display frame
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
    df = soccer_tracker.save_data(SAVE_CSV_PATH)
    
    print(f"Processing complete! Average FPS: {soccer_tracker.get_average_fps():.2f}")
    print(f"Output video saved to {OUTPUT_VIDEO_PATH}")
    print(f"Tracking data saved to {SAVE_CSV_PATH}")


if __name__ == "__main__":
    main()