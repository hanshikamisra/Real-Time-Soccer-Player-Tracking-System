# Soccer Tracking & Performance Analysis with YOLOv5 + DeepSORT

A real-time soccer player and ball tracking system using **YOLOv5** for object detection and **DeepSORT** for multi-object tracking. This project also includes speed estimation, heatmap visualization, and Voronoi diagram generation to assist in tactical performance analysis.

---

## Features

- **Real-time Object Detection** with custom-trained YOLOv5
- **Multi-Object Tracking** using DeepSORT with consistent Player IDs
- **Speed Estimation** based on pixel displacement and frame rate
- **Heatmap Visualization** of player movement density
- **Voronoi Diagrams** to analyze spatial dominance
- **Exportable CSV tracking data** for further analytics

---

## 🎯 Project Goals

- Track players, referees, and ball accurately from match footage
- Estimate player movement speeds in real-time
- Visualize tactical control and player positions across the pitch
- Provide a lightweight solution runnable on CPU with high accuracy

---

## 🖼️ Example Outputs

| Detection + Tracking | Heatmap | Voronoi |
|----------------------|---------|---------|
| ![Tracking](assets/tracking_demo.gif) | ![Heatmap](assets/heatmap.png) | ![Voronoi](assets/voronoi.png) |

---

## 🚀 How It Works

### 1. **Detection with YOLOv5**
- Detects 3 classes: **players**, **referees**, and **ball**
- Custom-trained on labeled soccer footage
- High-speed inference optimized for CPU/GPU

### 2. **Tracking with DeepSORT**
- Maintains unique IDs across frames using:
  - Kalman Filter (motion prediction)
  - Cosine distance (appearance features)
- Handles occlusion and re-identification

### 3. **Speed Estimation**
- Calculates displacement across frames:
  \[
  \text{Speed} = \frac{\sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}}{\Delta t}
  \]
- Frame-rate dependent (e.g., 30 FPS)
- Optional pixel-to-meter scaling

### 4. **Heatmaps**
- Uses `seaborn.kdeplot()` to visualize player movement density

### 5. **Voronoi Diagrams**
- Calculates zones of influence using `scipy.spatial.Voronoi`
- Visualizes team shape, coverage, and player spacing

---

## 📁 Directory Structure


## Setup

1. Clone the repo & create a virtual environment:
   ```bash
   git clone <your-repo-url>
   cd Soccer_tracking
   python -m venv venv
   venv\Scripts\activate  # on Windows

   
