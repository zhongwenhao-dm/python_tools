import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from matplotlib.animation import FuncAnimation

forward = np.array([0, 1, 0])  # 假设初始朝向为 y 轴正方向

def plot_trajectory_with_directions(positions, quats, step=30):
    # 转为方向向量
    dirs = R.from_quat(quats).apply(forward)  # 轨迹方向（以x轴为参考）

    # 3D 可视化
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], color='blue', label="Trajectory")

    # 每隔 N 个点显示方向箭头
    for i in range(0, len(positions), step):
        p = positions[i]
        d = dirs[i] * 2  # 可缩放箭头长度
        ax.quiver(p[0], p[1], p[2], d[0], d[1], d[2], color='red', linewidth=1)

    ax.set_title("3D Trajectory with Pose Directions")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.tight_layout()
    plt.show()

def play_trajectory_by_animation(timestamps, positions, quats, interval_ms=50):
    """动态播放轨迹动画"""
    print(f"数据长度: {len(positions)}")
    print(f"位置范围: X({np.min(positions[:, 0]):.2f}, {np.max(positions[:, 0]):.2f}), "
          f"Y({np.min(positions[:, 1]):.2f}, {np.max(positions[:, 1]):.2f}), "
          f"Z({np.min(positions[:, 2]):.2f}, {np.max(positions[:, 2]):.2f})")
    
    # 转换四元数为方向向量
    dirs = R.from_quat(quats).apply(forward)  # 姿态朝向单位向量

    # 创建图形
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 设置坐标轴范围，添加一些边距
    x_range = np.max(positions[:, 0]) - np.min(positions[:, 0])
    y_range = np.max(positions[:, 1]) - np.min(positions[:, 1])
    z_range = np.max(positions[:, 2]) - np.min(positions[:, 2])
    
    margin = 0.1
    ax.set_xlim(np.min(positions[:, 0]) - x_range * margin, 
                np.max(positions[:, 0]) + x_range * margin)
    ax.set_ylim(np.min(positions[:, 1]) - y_range * margin, 
                np.max(positions[:, 1]) + y_range * margin)
    ax.set_zlim(np.min(positions[:, 2]) - z_range * margin, 
                np.max(positions[:, 2]) + z_range * margin)
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Trajectory Animation")

    # 初始化绘图元素
    trajectory_line, = ax.plot([], [], [], 'b-', linewidth=2, label='Trajectory')
    point_marker, = ax.plot([], [], [], 'ro', markersize=8, label='Current Position')
    ax.legend()

    # 按照时间戳构造动画帧索引
    total_time = timestamps[-1] - timestamps[0]
    interval_sec = interval_ms / 1000.0
    frame_times = np.arange(0, total_time + interval_sec, interval_sec) + timestamps[0]
    frame_indices = np.searchsorted(timestamps, frame_times, side='right') - 1
    frame_indices = np.clip(frame_indices, 0, len(positions) - 1)

    print(f"总时间: {total_time:.2f}s")
    print(f"目标FPS: {1000 / interval_ms:.1f}")
    print(f"动画帧数: {len(frame_indices)}")

    def update(frame):
        i = frame_indices[frame]

        trajectory_line.set_data(positions[:i+1, 0], positions[:i+1, 1])
        trajectory_line.set_3d_properties(positions[:i+1, 2])

        p = positions[i]
        point_marker.set_data([p[0]], [p[1]])
        point_marker.set_3d_properties([p[2]])

        d = dirs[i] * (max(x_range, y_range, z_range) * 0.1)

        # 删除旧箭头
        for collection in ax.collections[:]:
            if hasattr(collection, '_arrows'):
                collection.remove()

        arrow = ax.quiver(p[0], p[1], p[2], d[0], d[1], d[2], 
                          color='red', arrow_length_ratio=0.1, linewidth=2)
        arrow._arrows = True

        progress = (timestamps[i] - timestamps[0]) / total_time * 100
        ax.set_title(f"3D Trajectory Animation - Progress: {progress:.1f}%")

        return trajectory_line, point_marker
  
    print("开始播放动画...")
    
    # 创建动画
    ani = FuncAnimation(fig, update, frames=len(frame_indices), interval=interval_ms, 
                       blit=False, repeat=True, cache_frame_data=False)
    
    plt.tight_layout()
    plt.show()
    
    return ani  # 返回动画对象以防止被垃圾回收

def play_trajectory_simple(timestamps, positions, quats, interval_ms=50):
    """简化版动画播放（无箭头）"""
    print(f"播放简化版动画，数据点数: {len(positions)}")
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 设置坐标轴
    ax.set_xlim(np.min(positions[:, 0]), np.max(positions[:, 0]))
    ax.set_ylim(np.min(positions[:, 1]), np.max(positions[:, 1]))
    ax.set_zlim(np.min(positions[:, 2]), np.max(positions[:, 2]))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Trajectory Animation (Simple)")

    # 初始化绘图元素
    trajectory_line, = ax.plot([], [], [], 'b-', linewidth=2)
    point_marker, = ax.plot([], [], [], 'ro', markersize=8)

    # 计算采样步长
    sampling_step = max(1, len(positions) // 200)  # 限制最多200帧
    
    def update(frame):
        i = min(frame * sampling_step, len(positions) - 1)
        
        # 更新轨迹
        trajectory_line.set_data(positions[:i+1, 0], positions[:i+1, 1])
        trajectory_line.set_3d_properties(positions[:i+1, 2])
        
        # 更新当前点
        p = positions[i]
        point_marker.set_data([p[0]], [p[1]])
        point_marker.set_3d_properties([p[2]])
        
        return trajectory_line, point_marker

    frame_count = (len(positions) + sampling_step - 1) // sampling_step
    ani = FuncAnimation(fig, update, frames=frame_count, interval=interval_ms, 
                       blit=False, repeat=True)
    
    plt.show()
    return ani

if __name__ == "__main__":
    # 读取 pose.txt
    pose_file = "/home/dmgz/ZWH/python_tools/trajectory_generation/pose.txt"
    pose = np.loadtxt(pose_file, delimiter=",", skiprows=1)
    timestamps = pose[:, 0]
    positions = pose[:, 1:4]
    quaternions = pose[:, 4:8]
    
    # 可视化轨迹和朝向
    plot_trajectory_with_directions(positions, quaternions, 20)
    
    # 动态播放轨迹
    # play_trajectory_by_animation(timestamps, positions, quaternions, interval_ms=50)
    # play_trajectory_simple(timestamps, positions, quaternions, interval_ms=50)
    