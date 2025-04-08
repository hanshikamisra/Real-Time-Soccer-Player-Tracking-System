import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# --- Field dimensions in meters (standard soccer field) ---
FIELD_WIDTH_M = 105
FIELD_HEIGHT_M = 68

class SpeedEstimator:
    def __init__(self, tracking_data_path, field_width_m=FIELD_WIDTH_M, field_height_m=FIELD_HEIGHT_M):
        """
        Initialize the speed estimator with tracking data
        
        Args:
            tracking_data_path: Path to the CSV file with tracking data
            field_width_m: Field width in meters
            field_height_m: Field height in meters
        """
        self.tracking_df = pd.read_csv(tracking_data_path)
        self.field_width_m = field_width_m
        self.field_height_m = field_height_m
        
        # Extract video frame dimensions (from max x and y coordinates)
        self.img_width = self.tracking_df['x'].max()
        self.img_height = self.tracking_df['y'].max()
        
        # Get video fps from frame differentials
        frame_ids = self.tracking_df['frame'].unique()
        if len(frame_ids) >= 2:
            # Estimate FPS from frame IDs (assuming sequential)
            self.fps = 30  # Default to 30 if cannot be determined
        else:
            self.fps = 30  # Default value
            
        # Calculate speed data
        self.speed_data = self.calculate_speeds()
        
    def pixels_to_meters(self, pixels):
        """Convert pixel distance to meters"""
        meters_per_pixel_x = self.field_width_m / self.img_width
        meters_per_pixel_y = self.field_height_m / self.img_height
        
        # Average of x and y scale factors
        meters_per_pixel = (meters_per_pixel_x + meters_per_pixel_y) / 2
        
        return pixels * meters_per_pixel
    
    def calculate_speeds(self):
        """Calculate speed for each track"""
        # Group by track_id
        groups = self.tracking_df.groupby('track_id')
        
        speed_data = []
        
        for track_id, group in groups:
            # Sort by frame
            group = group.sort_values('frame')
            
            # Skip if we have only one frame for this track
            if len(group) < 2:
                continue
                
            # Get position and frame data
            positions = group[['frame', 'x', 'y']].values
            
            # Calculate speeds
            for i in range(1, len(positions)):
                prev_frame, prev_x, prev_y = positions[i-1]
                curr_frame, curr_x, curr_y = positions[i]
                
                # Calculate distance in pixels
                pixel_distance = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                
                # Convert to meters
                meter_distance = self.pixels_to_meters(pixel_distance)
                
                # Calculate time difference (seconds)
                time_diff = (curr_frame - prev_frame) / self.fps
                
                # Calculate speed in m/s
                if time_diff > 0:
                    speed_mps = meter_distance / time_diff
                    
                    speed_data.append({
                        'track_id': track_id,
                        'frame': curr_frame,
                        'speed_mps': speed_mps,
                        'class': group.iloc[i]['class'] if 'class' in group.columns else None,
                        'x': curr_x,
                        'y': curr_y
                    })
        
        return pd.DataFrame(speed_data)
    
    def apply_smoothing(self, window_size=5):
        """Apply rolling average smoothing to speed data"""
        # Group by track_id
        groups = self.speed_data.groupby('track_id')
        
        smoothed_data = []
        
        for track_id, group in groups:
            # Sort by frame
            group = group.sort_values('frame')
            
            # Apply rolling average
            group['smooth_speed'] = group['speed_mps'].rolling(window=window_size, min_periods=1).mean()
            
            smoothed_data.append(group)
        
        # Combine all data
        self.speed_data = pd.concat(smoothed_data)
        return self.speed_data
    
    def save_speed_data(self, output_path):
        """Save speed data to CSV"""
        self.speed_data.to_csv(output_path, index=False)
        print(f"Speed data saved to {output_path}")
        return self.speed_data
    
    def get_max_speeds(self):
        """Get maximum speed for each track"""
        if self.speed_data.empty:
            return pd.DataFrame()
            
        # Group by track_id and get max speed
        max_speeds = self.speed_data.groupby('track_id')['speed_mps'].max().reset_index()
        max_speeds = max_speeds.sort_values('speed_mps', ascending=False)
        
        return max_speeds
    
    def get_average_speeds(self):
        """Get average speed for each track"""
        if self.speed_data.empty:
            return pd.DataFrame()
            
        # Group by track_id and get average speed
        avg_speeds = self.speed_data.groupby('track_id')['speed_mps'].mean().reset_index()
        avg_speeds = avg_speeds.sort_values('speed_mps', ascending=False)
        
        return avg_speeds
    
    def plot_speed_over_time(self, track_ids=None, output_path=None):
        """Plot speed over time for specific tracks"""
        if self.speed_data.empty:
            print("No speed data available")
            return
            
        plt.figure(figsize=(12, 6))
        
        if track_ids is None:
            # If no track_ids specified, get top 5 by max speed
            max_speeds = self.get_max_speeds()
            track_ids = max_speeds.head(5)['track_id'].values
        
        for track_id in track_ids:
            track_data = self.speed_data[self.speed_data['track_id'] == track_id]
            if not track_data.empty:
                plt.plot(track_data['frame'], track_data['speed_mps'], label=f'Player {track_id}')
        
        plt.title('Player Speed Over Time')
        plt.xlabel('Frame')
        plt.ylabel('Speed (m/s)')
        plt.legend()
        plt.grid(True)
        
        if output_path:
            plt.savefig(output_path)
            print(f"Speed plot saved to {output_path}")
        else:
            plt.show()
    
    def visualize_speed_on_video(self, original_tracking_path, output_path):
        """Create a new tracking CSV with speed data for visualization"""
        # Read original tracking data
        original_df = pd.read_csv(original_tracking_path)
        
        # Merge with speed data
        merged_df = pd.merge(
            original_df,
            self.speed_data[['track_id', 'frame', 'speed_mps']],
            on=['track_id', 'frame'],
            how='left'
        )
        
        # Fill missing speeds with 0
        merged_df['speed_mps'] = merged_df['speed_mps'].fillna(0)
        
        # Save to new CSV
        merged_df.to_csv(output_path, index=False)
        print(f"Tracking data with speed information saved to {output_path}")
        return merged_df


def main():
    # Set paths
    ROOT_DIR = os.getcwd()
    OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    TRACKING_DATA_PATH = os.path.join(OUTPUT_DIR, "tracking_data.csv")
    SPEED_DATA_PATH = os.path.join(OUTPUT_DIR, "speed_data.csv")
    SPEED_PLOT_PATH = os.path.join(OUTPUT_DIR, "speed_plot.png")
    TRACKING_WITH_SPEED_PATH = os.path.join(OUTPUT_DIR, "tracking_with_speed.csv")
    
    # Check if tracking data exists
    if not os.path.exists(TRACKING_DATA_PATH):
        print(f"Tracking data file not found: {TRACKING_DATA_PATH}")
        return
    
    # Initialize speed estimator
    speed_estimator = SpeedEstimator(TRACKING_DATA_PATH)
    
    # Apply smoothing
    speed_estimator.apply_smoothing(window_size=5)
    
    # Save speed data
    speed_estimator.save_speed_data(SPEED_DATA_PATH)
    
    # Plot speed over time
    speed_estimator.plot_speed_over_time(output_path=SPEED_PLOT_PATH)
    
    # Create tracking data with speed for visualization
    speed_estimator.visualize_speed_on_video(TRACKING_DATA_PATH, TRACKING_WITH_SPEED_PATH)
    
    # Print max speeds
    max_speeds = speed_estimator.get_max_speeds()
    print("\nTop 5 Players by Maximum Speed:")
    print(max_speeds.head(5))
    
    # Print average speeds
    avg_speeds = speed_estimator.get_average_speeds()
    print("\nTop 5 Players by Average Speed:")
    print(avg_speeds.head(5))


if __name__ == "__main__":
    main()