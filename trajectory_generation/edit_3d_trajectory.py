import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import splprep, splev
import pandas as pd
import os
from matplotlib.widgets import TextBox, Button

class TrajectoryEditor3D:
    def __init__(self, csv_file, output_file="edited_trajectory_3d.csv"):
        self.csv_file = csv_file
        self.output_file = output_file
        self.original_points = None
        self.control_points = None
        self.smoothed_trajectory = None
        self.selected_point_idx = None
        self.text_box = None
        self.select_button = None
        
        # Load data
        self.load_trajectory()
        
        # Create GUI
        self.setup_plot()
        
    def load_trajectory(self):
        """Load 3D trajectory points from CSV file"""
        try:
            if self.csv_file.endswith('.csv'):
                data = pd.read_csv(self.csv_file)
                if data.shape[1] >= 3:
                    self.original_points = data.iloc[:, :3].values
                elif data.shape[1] == 2:
                    # If 2D data, add z=0 dimension
                    xy_data = data.iloc[:, :2].values
                    self.original_points = np.column_stack([xy_data, np.zeros(len(xy_data))])
                else:
                    raise ValueError("CSV file needs at least 2 columns")
            else:
                # Try space-separated format
                self.original_points = np.loadtxt(self.csv_file)
                
            # Create editable control points (copy of original points)
            self.control_points = self.original_points.copy()
            
            print(f"Successfully loaded {len(self.original_points)} 3D points")
            print(f"Data range: X[{self.original_points[:, 0].min():.2f}, {self.original_points[:, 0].max():.2f}], "
                  f"Y[{self.original_points[:, 1].min():.2f}, {self.original_points[:, 1].max():.2f}], "
                  f"Z[{self.original_points[:, 2].min():.2f}, {self.original_points[:, 2].max():.2f}]")
            
        except Exception as e:
            print(f"Error loading file: {e}")
            # Create sample data if loading fails
            self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample 3D data"""
        t = np.linspace(0, 4*np.pi, 15)
        x = np.cos(t) * 2
        y = np.sin(t) * 2  
        z = t / 2
        self.original_points = np.column_stack([x, y, z])
        self.control_points = self.original_points.copy()
        print("Using sample spiral data")
    
    def smooth_trajectory_3d(self, points, num_points=300, smoothness=2.0):
        """Smooth 3D trajectory"""
        if len(points) < 4:
            return points
            
        try:
            points = np.array(points)
            x, y, z = points[:, 0], points[:, 1], points[:, 2]
            
            # Fit 3D B-spline curve
            tck, u = splprep([x, y, z], s=smoothness, k=min(3, len(points)-1))
            u_fine = np.linspace(0, 1, num_points)
            x_fine, y_fine, z_fine = splev(u_fine, tck)
            smoothed = np.vstack((x_fine, y_fine, z_fine)).T
            return smoothed
        except Exception as e:
            print(f"Smoothing error: {e}")
            return points
    
    def setup_plot(self):
        """Setup 3D GUI"""
        self.fig = plt.figure(figsize=(16, 10))
        
        # Adjust layout for input box
        plt.subplots_adjust(bottom=0.15, right=0.85)
        
        # Create main 3D plot area
        self.ax = self.fig.add_subplot(121, projection='3d')
        
        # Create control panel area
        self.control_ax = self.fig.add_subplot(122)
        self.control_ax.set_xlim(0, 1)
        self.control_ax.set_ylim(0, 1)
        self.control_ax.axis('off')
        
        # Set title
        self.fig.suptitle('3D Trajectory Editor - Input Box Selection', fontsize=16, fontweight='bold')
        
        # Create input box and buttons
        self.create_input_widgets()
        
        # Create control panel
        self.create_control_panel()
        
        # Bind keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        
        # Initialize plot
        self.update_plot()
        
        # Show instructions
        self.show_instructions()
    
    def create_input_widgets(self):
        """Create input box and selection buttons"""
        # Create input box - make it wider to accommodate more digits
        ax_textbox = plt.axes([0.1, 0.05, 0.15, 0.04])  # [left, bottom, width, height]
        self.text_box = TextBox(ax_textbox, 'Point Index:', initial="0")
        self.text_box.on_submit(self.on_text_submit)
        
        # Create select button
        ax_button = plt.axes([0.27, 0.05, 0.08, 0.04])
        self.select_button = Button(ax_button, 'Select Point')
        self.select_button.on_clicked(self.on_button_click)
        
        # Create clear selection button
        ax_clear = plt.axes([0.36, 0.05, 0.08, 0.04])
        self.clear_button = Button(ax_clear, 'Clear Selection')
        self.clear_button.on_clicked(self.on_clear_click)
    
    def on_text_submit(self, text):
        """Text box submit event"""
        self.select_point_by_text(text)
    
    def on_button_click(self, event):
        """Select button click event"""
        text = self.text_box.text
        self.select_point_by_text(text)
    
    def on_clear_click(self, event):
        """Clear selection button click event"""
        self.selected_point_idx = None
        self.text_box.set_val("")
        print("Selection cleared")
        self.update_plot()
    
    def select_point_by_text(self, text):
        """Select point by text input"""
        if not text.strip():
            print("Please enter point index")
            return
            
        try:
            point_idx = int(text)
            
            if 0 <= point_idx < len(self.control_points):
                self.selected_point_idx = point_idx
                print(f"Selected point {point_idx}: {self.control_points[point_idx]}")
                self.update_plot()
            else:
                print(f"Point {point_idx} does not exist (total points: {len(self.control_points)})")
                print(f"Valid range: 0-{len(self.control_points)-1}")
                
        except ValueError:
            print("Please enter a valid number")
    
    def create_control_panel(self):
        """Create control panel"""
        # Add text information in control panel
        self.info_text = self.control_ax.text(0.05, 0.95, "", fontsize=10, 
                                            verticalalignment='top', fontfamily='monospace')
        
        # Add instructions
        instructions = """CONTROL PANEL

Point Selection:
1. Input Box: Enter point index (0 to {max_points})
2. PageUp/PageDown: Cycle through points
3. Home/End: Select first/last point

Point Movement:
• Arrow Keys: Move selected point
  ↑/↓: Z-axis  ←/→: X-axis
• Shift + Arrow Keys: Y-axis movement
• +/-: Adjust movement step size

Other Controls:
• R: Reset • S: Save • C: Clear Selection

Current Status:
""".format(max_points=len(self.control_points)-1 if self.control_points is not None else 0)
        
        self.control_ax.text(0.05, 0.75, instructions, fontsize=9, 
                           verticalalignment='top', fontfamily='monospace')
    
    def show_instructions(self):
        """Show operation instructions"""
        print("\n" + "="*60)
        print("3D Trajectory Editor - Instructions")
        print("="*60)
        print("Main Feature: Use input box to enter point index directly!")
        print("")
        print("Point Selection Methods:")
        print("• Input Box: Enter any number (0 to {})".format(len(self.control_points)-1))
        print("• PageUp/PageDown: Cycle through all points")
        print("• Home/End: Select first/last point")
        print("")
        print("Point Movement Controls:")
        print("• Arrow Keys: Move selected point")
        print("• Shift + Arrow Keys: Y-axis movement")
        print("• +/-: Fine/coarse adjustment of movement step")
        print("")
        print("Other Controls:")
        print("• R: Reset all points to original positions")
        print("• S: Save current trajectory")
        print("• C: Clear current selection")
        print("="*60)
        
        if len(self.control_points) > 0:
            print(f"Loaded {len(self.control_points)} control points")
            print(f"Valid index range: 0-{len(self.control_points)-1}")
            print("Try entering a point index in the input box!")
    
    def update_plot(self):
        """Update 3D plot display"""
        self.ax.clear()
        
        # Recalculate smoothed trajectory
        self.smoothed_trajectory = self.smooth_trajectory_3d(self.control_points, 
                                                           num_points=300, 
                                                           smoothness=2.0)
        
        # Draw original trajectory (gray thin line)
        self.ax.plot(self.original_points[:, 0], 
                    self.original_points[:, 1], 
                    self.original_points[:, 2], 
                    'gray', alpha=0.5, linewidth=1, label='Original')
        
        # Draw smoothed trajectory (blue thick line)
        if self.smoothed_trajectory is not None:
            self.ax.plot(self.smoothed_trajectory[:, 0], 
                        self.smoothed_trajectory[:, 1], 
                        self.smoothed_trajectory[:, 2], 
                        'b-', linewidth=3, alpha=0.8, label='Smoothed')
        
        # Draw control points
        if len(self.control_points) > 0:
            colors = []
            sizes = []
            for i in range(len(self.control_points)):
                if i == self.selected_point_idx:
                    colors.append('red')
                    sizes.append(120)
                else:
                    colors.append('orange')
                    sizes.append(80)
            
            self.ax.scatter(self.control_points[:, 0], 
                          self.control_points[:, 1], 
                          self.control_points[:, 2], 
                          c=colors, s=sizes, alpha=0.9, 
                          edgecolors='black', linewidth=1.5)
            
            # Add numbers to all points
            for i, point in enumerate(self.control_points):
                color = 'red' if i == self.selected_point_idx else 'darkblue'
                weight = 'bold' if i == self.selected_point_idx else 'normal'
                
                label = str(i)
                self.ax.text(point[0], point[1], point[2], f' {label}',
                           fontsize=8, color=color, weight=weight)
            
            # Add detailed information for selected point
            if self.selected_point_idx is not None:
                point = self.control_points[self.selected_point_idx]
                self.ax.text(point[0], point[1], point[2], 
                           f'\nPoint{self.selected_point_idx}\nX:{point[0]:.2f} Y:{point[1]:.2f} Z:{point[2]:.2f}',
                           fontsize=9, color='red', weight='bold')
        
        # Set axis labels and title
        self.ax.set_xlabel('X Axis')
        self.ax.set_ylabel('Y Axis')
        self.ax.set_zlabel('Z Axis')
        title = f'Control Points: {len(self.control_points)}'
        if self.selected_point_idx is not None:
            title += f' | Selected: Point{self.selected_point_idx}'
        self.ax.set_title(title)
        
        # Add legend
        self.ax.legend()
        
        # Set equal axis scale
        self.set_axes_equal()
        
        # Update control panel information
        self.update_control_panel()
        
        # Refresh display
        self.fig.canvas.draw()
    
    def update_control_panel(self):
        """Update control panel information"""
        status = f"Total Points: {len(self.control_points)}\n"
        if self.selected_point_idx is not None:
            point = self.control_points[self.selected_point_idx]
            status += f"Selected: Point {self.selected_point_idx}\n"
            status += f"Position: X:{point[0]:.3f} Y:{point[1]:.3f} Z:{point[2]:.3f}\n"
        else:
            status += "Selected: None\n"
            status += "Enter point index in input box\n"
        
        status += f"\nMove Step: {getattr(self, 'move_step', 0.1):.3f}\n"
        
        # Show selection hint
        if len(self.control_points) > 0:
            status += f"\nValid range: 0-{len(self.control_points)-1}"
        
        self.info_text.set_text(status)
    
    def set_axes_equal(self):
        """Set 3D axis equal scale"""
        if len(self.control_points) == 0:
            return
            
        # Get data range
        all_points = np.vstack([self.control_points, self.smoothed_trajectory])
        max_range = np.array([all_points[:, 0].max() - all_points[:, 0].min(),
                             all_points[:, 1].max() - all_points[:, 1].min(),
                             all_points[:, 2].max() - all_points[:, 2].min()]).max() / 2.0
        
        mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
        mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
        mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5
        
        self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    def on_key_press(self, event):
        """Keyboard event handling"""
        if event.key is None:
            return
            
        key = event.key.lower()
        
        # Navigation key selection (ONLY KEEP THESE)
        if key == 'pageup':
            if len(self.control_points) > 0:
                if self.selected_point_idx is None:
                    self.selected_point_idx = 0
                else:
                    self.selected_point_idx = (self.selected_point_idx - 1) % len(self.control_points)
                self.update_textbox_from_selection()
                self.update_plot()
            return
            
        elif key == 'pagedown':
            if len(self.control_points) > 0:
                if self.selected_point_idx is None:
                    self.selected_point_idx = 0
                else:
                    self.selected_point_idx = (self.selected_point_idx + 1) % len(self.control_points)
                self.update_textbox_from_selection()
                self.update_plot()
            return
            
        elif key == 'home':
            if len(self.control_points) > 0:
                self.selected_point_idx = 0
                self.update_textbox_from_selection()
                self.update_plot()
            return
            
        elif key == 'end':
            if len(self.control_points) > 0:
                self.selected_point_idx = len(self.control_points) - 1
                self.update_textbox_from_selection()
                self.update_plot()
            return
        
        # Clear selection
        if key == 'c':
            self.selected_point_idx = None
            self.text_box.set_val("")
            print("Selection cleared")
            self.update_plot()
            return
        
        # Need selected point for movement operations
        if self.selected_point_idx is None:
            if key in ['up', 'down', 'left', 'right']:
                print("Please select a point first (use input box or PageUp/PageDown)")
            return
        
        # Set movement step size
        if not hasattr(self, 'move_step'):
            self.move_step = 0.1
        
        # Adjust movement step size
        if key == '+' or key == '=':
            self.move_step = max(0.001, self.move_step / 2)
            print(f"Move step: {self.move_step:.3f} (finer)")
            self.update_plot()
            return
        elif key == '-':
            self.move_step = min(1.0, self.move_step * 2)
            print(f"Move step: {self.move_step:.3f} (coarser)")
            self.update_plot()
            return
        
        # Move control point
        moved = False
        if key == 'up':
            if hasattr(event, 'shift') and event.shift:
                self.control_points[self.selected_point_idx][1] += self.move_step  # Y+
            else:
                self.control_points[self.selected_point_idx][2] += self.move_step  # Z+
            moved = True
        elif key == 'down':
            if hasattr(event, 'shift') and event.shift:
                self.control_points[self.selected_point_idx][1] -= self.move_step  # Y-
            else:
                self.control_points[self.selected_point_idx][2] -= self.move_step  # Z-
            moved = True
        elif key == 'left':
            self.control_points[self.selected_point_idx][0] -= self.move_step  # X-
            moved = True
        elif key == 'right':
            self.control_points[self.selected_point_idx][0] += self.move_step  # X+
            moved = True
        
        # Other shortcuts
        if key == 'r':
            self.control_points = self.original_points.copy()
            print("Reset to original positions")
            moved = True
        elif key == 's':
            self.save_trajectory()
            print("Trajectory saved")
        
        if moved:
            if self.selected_point_idx is not None:
                point = self.control_points[self.selected_point_idx]
                print(f"Point {self.selected_point_idx} moved to: ({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})")
            self.update_plot()
    
    def update_textbox_from_selection(self):
        """Update input box based on current selection"""
        if self.selected_point_idx is not None:
            self.text_box.set_val(str(self.selected_point_idx))
    
    def on_close(self, event):
        """Window close event - save results"""
        self.save_trajectory()
    
    def save_trajectory(self):
        """Save edited trajectory"""
        try:
            # Save final smoothed trajectory
            if self.smoothed_trajectory is not None:
                np.savetxt(self.output_file, self.smoothed_trajectory, 
                          delimiter=',', header='x,y,z', comments='', fmt='%.6f')
                print(f"Edited 3D trajectory saved to: {self.output_file}")
                print(f"Number of trajectory points: {len(self.smoothed_trajectory)}")
            else:
                print("No trajectory data to save")
                
        except Exception as e:
            print(f"Error saving file: {e}")
    
    def show(self):
        """Show editor interface"""
        plt.show()

def main(csv_file="trajectory_3d.csv", output_file="edited_trajectory_3d.csv"):
    """Main function"""
    print("Starting 3D Trajectory Editor...")
    print(f"Input file: {csv_file}")
    print(f"Output file: {output_file}")
    
    # Check if input file exists
    if not os.path.exists(csv_file):
        print(f"Warning: File {csv_file} does not exist, using sample data")
    
    # Create editor instance and show
    editor = TrajectoryEditor3D(csv_file, output_file)
    editor.show()

if __name__ == "__main__":
    # Use your generated CSV file
    main("trajectory_3d.csv", "edited_trajectory_3d.csv")