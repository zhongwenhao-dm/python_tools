import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter

def load_pose_file(file_path):
    """加载位姿文件"""
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    timestamps = data[:, 0]
    positions = data[:, 1:4]  # tx, ty, tz
    quaternions = data[:, 4:]  # qx, qy, qz, qw
    return timestamps, positions, quaternions

def check_rotation_quality(quaternions, timestamps):
    """检查旋转质量并返回问题指标"""
    # 确保四元数归一化
    norms = np.linalg.norm(quaternions, axis=1)
    norm_deviation = np.abs(norms - 1.0)
    
    # 计算角速度
    rotations = R.from_quat(quaternions)
    angular_rates = np.zeros((len(rotations)-1, 3))
    
    for i in range(1, len(rotations)):
        delta_rot = rotations[i-1].inv() * rotations[i]
        angular_rates[i-1] = delta_rot.as_rotvec() / (timestamps[i] - timestamps[i-1])
        if np.linalg.norm(angular_rates[i-1]) > 5:
            print(f"High angular rate detected at frame {i-1}: {angular_rates[i-1]}")
    
    # 计算角加速度
    angular_acc = np.zeros((len(angular_rates)-1, 3))
    for i in range(1, len(angular_rates)):
        angular_acc[i-1] = (angular_rates[i] - angular_rates[i-1]) / (timestamps[i+1] - timestamps[i])
    
    # 检测异常指标
    problems = {
        'non_unit_quat': np.sum(norm_deviation > 0.01),
        'high_angular_rate': np.sum(np.linalg.norm(angular_rates, axis=1) > 5),  # rad/s
        'high_angular_acc': np.sum(np.linalg.norm(angular_acc, axis=1) > 10),    # rad/s²
        'discontinuities': 0
    }
    
    # 检测四元数不连续（符号翻转）
    quat_dots = np.sum(quaternions[1:] * quaternions[:-1], axis=1)
    problems['discontinuities'] = np.sum(quat_dots < 0)
    
    return {
        'norms': norms,
        'angular_rates': angular_rates,
        'angular_acc': angular_acc,
        'problems': problems,
        'quat_dots': quat_dots
    }

def plot_rotation_analysis(results, timestamps):
    """绘制旋转分析结果"""
    plt.figure(figsize=(15, 10))
    
    # 四元数范数
    plt.subplot(3, 1, 1)
    plt.plot(timestamps, results['norms'], 'b-')
    plt.axhline(1.0, color='r', linestyle='--')
    plt.ylabel('Quaternion Norm')
    plt.title('Rotation Quality Analysis')
    
    # 角速度
    plt.subplot(3, 1, 2)
    angular_rate_norms = np.linalg.norm(results['angular_rates'], axis=1)
    plt.plot(timestamps[1:], angular_rate_norms, 'g-')
    plt.axhline(5.0, color='r', linestyle='--', label='Threshold (5 rad/s)')
    plt.ylabel('Angular Rate (rad/s)')
    plt.legend()
    
    # 角加速度
    plt.subplot(3, 1, 3)
    angular_acc_norms = np.linalg.norm(results['angular_acc'], axis=1)
    plt.plot(timestamps[2:], angular_acc_norms, 'r-')
    plt.axhline(10.0, color='r', linestyle='--', label='Threshold (10 rad/s²)')
    plt.ylabel('Angular Acceleration (rad/s²)')
    plt.xlabel('Time (s)')
    plt.legend()
    
    plt.tight_layout()
    
    # 打印问题摘要
    print("\n=== Rotation Quality Report ===")
    print(f"Non-unit quaternions: {results['problems']['non_unit_quat']} frames")
    print(f"High angular rate (>5 rad/s): {results['problems']['high_angular_rate']} frames")
    print(f"High angular acceleration (>10 rad/s²): {results['problems']['high_angular_acc']} frames")
    print(f"Quaternion discontinuities: {results['problems']['discontinuities']} frames")
    
    if any(v > 0 for v in results['problems'].values()):
        print("\nWARNING: Potential rotation quality issues detected!")
    else:
        print("\nRotation quality is good!")

def smooth_rotations(quaternions, window_length=5, polyorder=2):
    """使用Savitzky-Golay滤波器平滑四元数"""
    # 确保四元数连续（无符号翻转）
    for i in range(1, len(quaternions)):
        if np.dot(quaternions[i], quaternions[i-1]) < 0:
            quaternions[i] = -quaternions[i]
    
    # 对每个四元数分量单独应用滤波器
    smoothed = np.zeros_like(quaternions)
    for i in range(4):
        smoothed[:, i] = savgol_filter(quaternions[:, i], window_length, polyorder)
    
    # 重新归一化
    norms = np.linalg.norm(smoothed, axis=1)
    smoothed = smoothed / norms[:, np.newaxis]
    
    return smoothed

def animate_trajectory_with_checks(timestamps, positions, quaternions, interval=50):
    """带旋转检查的动画播放"""
    # 先分析旋转质量
    rotation_results = check_rotation_quality(quaternions, timestamps)
    plot_rotation_analysis(rotation_results, timestamps)
    
    # 如果有问题，应用平滑
    # if any(v > 0 for v in rotation_results['problems'].values()):
    #     print("\nApplying smoothing to quaternions...")
    #     quaternions = smooth_rotations(quaternions)
        
    #     # 重新检查平滑后的质量
    #     rotation_results = check_rotation_quality(quaternions, timestamps)
    #     plot_rotation_analysis(rotation_results, timestamps)
    
    # 创建动画可视化
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制完整轨迹
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', alpha=0.3, label='Trajectory')
    
    # 初始化当前位姿表示
    current_pos, = ax.plot([], [], [], 'ro', markersize=8, label='Current Position')
    
    # 初始化坐标系箭头
    arrow_length = 5  # 箭头长度(米)
    x_arrow = ax.quiver([], [], [], [], [], [], color='r', arrow_length_ratio=0.1, label='East')
    y_arrow = ax.quiver([], [], [], [], [], [], color='g', arrow_length_ratio=0.1, label='North')
    z_arrow = ax.quiver([], [], [], [], [], [], color='b', arrow_length_ratio=0.1, label='Up')
    
    # 设置ENU坐标系标签和范围
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_zlabel('Up (m)')
    ax.set_title('Trajectory Playback with Rotation Check (ENU)')
    
    # 设置等比例轴
    max_range = np.array([positions[:, 0].max()-positions[:, 0].min(), 
                          positions[:, 1].max()-positions[:, 1].min(), 
                          positions[:, 2].max()-positions[:, 2].min()]).max() / 2.0
    mid_x = (positions[:, 0].max()+positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max()+positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max()+positions[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.legend()
    
    def update(frame):
        # 更新当前位置
        current_pos.set_data([positions[frame, 0]], [positions[frame, 1]])
        current_pos.set_3d_properties([positions[frame, 2]])
        
        # 计算当前姿态的坐标系方向
        rotation = R.from_quat(quaternions[frame])
        x_dir = rotation.apply([1, 0, 0]) * arrow_length
        y_dir = rotation.apply([0, 1, 0]) * arrow_length
        z_dir = rotation.apply([0, 0, 1]) * arrow_length
        
        # 更新坐标系箭头
        x_arrow.set_segments(np.array([positions[frame], positions[frame] + x_dir]).reshape(1, 2, 3))
        y_arrow.set_segments(np.array([positions[frame], positions[frame] + y_dir]).reshape(1, 2, 3))
        z_arrow.set_segments(np.array([positions[frame], positions[frame] + z_dir]).reshape(1, 2, 3))
        
        # 显示当前时间和帧号
        ax.set_title(f'Trajectory Playback (ENU)\nTime: {timestamps[frame]:.2f}s | Frame: {frame}')
        
        return current_pos, x_arrow, y_arrow, z_arrow
    
    # 创建动画
    ani = FuncAnimation(fig, update, frames=len(positions), 
                        interval=1, blit=False, repeat=True)
    
    plt.tight_layout()
    plt.show()
    
    return ani

if __name__ == "__main__":
    # 加载位姿文件
    pose_file = "pose.txt"  # 替换为你的文件路径
    timestamps, positions, quaternions = load_pose_file(pose_file)
    
    print(f"Loaded {len(timestamps)} poses")
    print(f"Time duration: {timestamps[-1]:.2f} seconds")
    print(f"Position range: X({positions[:, 0].min():.2f}, {positions[:, 0].max():.2f})m "
          f"Y({positions[:, 1].min():.2f}, {positions[:, 1].max():.2f})m "
          f"Z({positions[:, 2].min():.2f}, {positions[:, 2].max():.2f})m")
    
    # 运行动画并检查旋转质量
    animate_trajectory_with_checks(timestamps, positions, quaternions, interval=10)