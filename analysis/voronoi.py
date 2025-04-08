import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import cv2

def generate_voronoi(csv_path, frame_id=None, output_dir=None, video_path=None, team_based=False):
    """
    Generate Voronoi diagram for player positions
    
    Args:
        csv_path: Path to tracking data CSV
        frame_id: Specific frame to visualize (None for all frames)
        output_dir: Directory to save output images
        video_path: Path to video file (to get dimensions)
        team_based: Whether to generate team-based Voronoi diagrams
    """
    print(f"Generating Voronoi diagrams from {csv_path}...")
    
    # Create output directory if needed
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tracking data
    df = pd.read_csv(csv_path)
    
    # Filter player class only (typically class 0)
    player_df = df[df['class'] == 0]
    
    # If video path provided, get dimensions
    img_width = 1280  # Default
    img_height = 720  # Default
    if video_path:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
    
    # Process specific frame if requested
    if frame_id is not None:
        frames_to_process = [frame_id]
    else:
        # Select a subset of frames to generate diagrams for
        # Too many frames would be overwhelming, so pick evenly spaced frames
        total_frames = player_df['frame'].max()
        frames_to_process = np.linspace(0, total_frames, 10, dtype=int)
    
    for frame in frames_to_process:
        frame_data = player_df[player_df['frame'] == frame]
        
        # We need at least 4 points for a meaningful Voronoi diagram
        if len(frame_data) < 4:
            print(f"Not enough player positions in frame {frame} for Voronoi diagram")
            continue
        
        # Generate basic Voronoi diagram
        points = frame_data[['x', 'y']].values
        
        # Compute Voronoi diagram
        vor = Voronoi(points)
        
        plt.figure(figsize=(12, 8))
        voronoi_plot_2d(vor, show_points=True, point_size=10)
        
        # Add player IDs to points
        for i, (x, y) in enumerate(points):
            track_id = frame_data.iloc[i]['track_id']
            plt.text(x+10, y+10, str(int(track_id)), fontsize=9)
        
        plt.title(f'Player Influence Zones (Frame {frame})')
        plt.xlim(0, img_width)
        plt.ylim(img_height, 0)  # Invert Y-axis to match image coordinates
        
        output_path = os.path.join(output_dir, f"voronoi_frame_{frame}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Voronoi diagram for frame {frame} saved to {output_path}")
        plt.close()
        
        # Generate team-based Voronoi if requested and team data available
        if team_based and 'team' in frame_data.columns:
            # Process each team separately
            for team_id in frame_data['team'].unique():
                team_data = frame_data[frame_data['team'] == team_id]
                
                # Need at least 3 points
                if len(team_data) < 3:
                    continue
                
                team_points = team_data[['x', 'y']].values
                
                # Color based on team
                color = 'red' if team_id == 0 else 'blue'
                
                plt.figure(figsize=(12, 8))
                plt.scatter(team_points[:, 0], team_points[:, 1], 
                           color=color, s=50, label=f'Team {team_id}')
                
                # Add player IDs
                for i, (x, y) in enumerate(team_points):
                    track_id = team_data.iloc[i]['track_id']
                    plt.text(x+10, y+10, str(int(track_id)), fontsize=9)
                
                # Draw team formation lines
                for i in range(len(team_points)):
                    for j in range(i+1, len(team_points)):
                        plt.plot([team_points[i, 0], team_points[j, 0]], 
                                [team_points[i, 1], team_points[j, 1]], 
                                color=color, alpha=0.3)
                
                plt.title(f'Team {team_id} Formation (Frame {frame})')
                plt.xlim(0, img_width)
                plt.ylim(img_height, 0)
                
                output_path = os.path.join(output_dir, f"team_{team_id}_formation_frame_{frame}.png")