import numpy as np
import open3d as o3d

import sys
sys.path.append('/home/dmgz/ZWH/python_tools')
from trajectory_generation.simulate_pose import smooth_direction_vectors, compute_smooth_quaternions
from trajectory_generation.validate_pose import plot_trajectory_with_directions, play_trajectory_by_animation


if __name__ == "__main__":
    repo_dir = "/home/dmgz/ZWH/python_tools/trail_extraction/"
    centerline_file_path = repo_dir + "data/fitted_curve.ply"
    output_pose_file = repo_dir + "data/pose.txt"
    
    pcd = o3d.io.read_point_cloud(centerline_file_path)
    positions = np.array([i for i in pcd.points])
    
    print("Computing smooth direction vectors...")
    directions = smooth_direction_vectors(positions, window_size=5)
    
    print("Computing smooth quaternions...")
    quaternions = compute_smooth_quaternions(directions)
    
    timestamps = np.arange(0, 20, 20/len(positions))[:, None]
    translation = np.array([7.59330649e+05, 2.52441620e+06, 5.71528876e+01]) + np.array([0.14558328375312524, -0.05647050717395352, 0.08209491721111642])
    pose = np.hstack([
        timestamps,
        positions + translation,
        np.asarray(quaternions)
    ])
    
    np.savetxt(output_pose_file, pose, fmt="%.6f", delimiter=",",
              header="timestamp,tx,ty,tz,qx,qy,qz,qw", comments='')
    
    # visualize
    # plot_trajectory_with_directions(positions, quaternions, 20)
    play_trajectory_by_animation(timestamps.flatten(), positions, quaternions, interval_ms=50)