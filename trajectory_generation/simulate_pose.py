import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

Rot = R.from_matrix(np.array([
    [-1, 0, 0],
    [ 0,-1, 0],
    [ 0, 0,-1]]))

def smooth_and_interpolate(trajectory, num_points=1000, smooth=0):
    # 拆分 x, y, z
    x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]

    # 拟合 B 样条曲线
    tck, u = splprep([x, y, z], s=smooth)

    # 生成均匀间隔的插值点
    u_fine = np.linspace(0, 1, num_points)
    x_new, y_new, z_new = splev(u_fine, tck)

    return np.vstack([x_new, y_new, z_new]).T
  
def plot_trajectory_with_labels(trajectory, label_interval=100):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b-', label='Interpolated Trajectory')

    for i in range(0, len(trajectory), label_interval):
        pt = trajectory[i]
        ax.text(pt[0], pt[1], pt[2], str(i), color='red', fontsize=8)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3D Trajectory with Index Labels")
    plt.legend()
    plt.tight_layout()
    plt.show()

def smooth_direction_vectors(positions, window_size=5):
    """
    计算平滑的方向向量
    """
    N = len(positions)
    directions = np.zeros_like(positions)
    
    for i in range(N):
        # 动态调整窗口大小
        left = max(0, i - window_size)
        right = min(N - 1, i + window_size)
        
        if right == left:
            # 只有一个点，使用前一个方向或默认方向
            if i > 0:
                directions[i] = directions[i-1]
            else:
                directions[i] = np.array([0, 1, 0])  # 默认朝向
        else:
            direction = positions[right] - positions[left]
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                directions[i] = direction / norm
            else:
                # 如果距离太小，使用前一个方向
                if i > 0:
                    directions[i] = directions[i-1]
                else:
                    directions[i] = np.array([0, 1, 0])
    
    return directions

def compute_smooth_quaternions(directions, initial_forward=np.array([0, 1, 0])):
    """
    计算平滑的四元数序列
    """
    N = len(directions)
    quaternions = []
    
    # 计算初始四元数
    q_prev = compute_rotation_quaternion(initial_forward, directions[0])
    quaternions.append(q_prev)
    
    for i in range(1, N):
        # 计算当前方向对应的四元数
        q_curr = compute_rotation_quaternion(initial_forward, directions[i], i)
        
        # 确保四元数连续性（选择最近的表示）
        if np.dot(q_prev, q_curr) < 0:
            q_curr = -q_curr
        
        quaternions.append(q_curr)
        q_prev = q_curr
    
    return np.array(quaternions)
  
def compute_no_roll_rotation(dir_vec, world_up=np.array([0, 0, 1])):
    dir_vec = dir_vec / np.linalg.norm(dir_vec)

    # 叉乘顺序不能反！！！！
    # 计算右向量
    right = np.cross(dir_vec, world_up)
    if np.linalg.norm(right) < 1e-6:
        # forward接近world_up或反向，选择另一个up
        alt_up = np.array([1, 0, 0])
        right = np.cross(alt_up, dir_vec)

    right = right / np.linalg.norm(right)

    # 重新计算up，保证正交
    up = np.cross(right, dir_vec)

    # 构造旋转矩阵：列向量为 right, up, forward
    rot_mat = np.column_stack((right, dir_vec, up))

    # 转成四元数
    quat = R.from_matrix(rot_mat).as_quat()

    return quat

def compute_rotation_quaternion(from_vec, to_vec, idx=-1, eps=1e-4):
    """
    计算从from_vec到to_vec的旋转四元数
    """
    from_vec = from_vec / np.linalg.norm(from_vec)
    to_vec = to_vec / np.linalg.norm(to_vec)
    
    dot_product = np.dot(from_vec, to_vec)
    
    # 处理平行情况
    if np.isclose(dot_product, 1.0, atol=eps):
        return np.array([0, 0, 0, 1])  # 无旋转
    
    # 处理反向情况
    if np.isclose(dot_product, -1.0, atol=eps):
        axis_candidates = [
            np.array([1, 0, 0]),  # x轴
            np.array([0, 0, 1]),  # z轴
            np.array([0, 1, 0])   # y轴作为最后选择
        ]
        
        for axis in axis_candidates:
            if abs(np.dot(from_vec, axis)) < 0.9:  # 不平行
                  # 施密特正交化
                  axis = axis - np.dot(axis, from_vec) * from_vec
                  axis = axis / np.linalg.norm(axis)
                  return R.from_rotvec(np.pi * axis).as_quat()
    
    quat = compute_no_roll_rotation(to_vec)
    
    return quat

def generate_pose(trajectory, speed_segments, output_file, frame_rate=30):
    N = len(trajectory)
    
    # ---------- 坐标系转换：以第一个点为原点 ----------
    origin = trajectory[0].copy()
    trajectory_enu = trajectory - origin  # 相对于第一个点的坐标
    
    # ---------- 计算累计距离 ----------
    cumulative_distance = np.zeros(N)
    for i in range(1, N):
        cumulative_distance[i] = cumulative_distance[i - 1] + np.linalg.norm(trajectory_enu[i] - trajectory_enu[i - 1])
        
    # ---------- 构建速度控制点 ----------
    speed_points_s = []
    speed_points_v = []

    for start, end, v_start, v_end in speed_segments:
        s_start = cumulative_distance[start]
        s_end = cumulative_distance[end-1]
        speed_points_s += [s_start, s_end]
        speed_points_v += [v_start / 3.6, v_end / 3.6]  # 转换为 m/s

    # ---------- 速度插值 ----------
    v_interp = PchipInterpolator(speed_points_s, speed_points_v)
    speeds = v_interp(cumulative_distance)

    # ---------- 计算时间戳 ----------
    min_speed = 1
    speeds = np.clip(speeds, min_speed, None)
    
    # ---------- 生成新的轨迹点 ----------
    delta_t = 1.0 / frame_rate  # 每帧的时间间隔
    timestamps = [0.0]
    new_positions = [trajectory_enu[0]]
    new_cumulative_distance = [0.0]
    new_speeds = [speeds[0]]
    
    current_distance = 0.0
    current_index = 0
    
    while current_index < N - 1:
        next_distance = current_distance + speeds[current_index] * delta_t
        if next_distance >= cumulative_distance[-1]:
            break
        
        # 找到下一个累计距离对应的索引
        next_index = np.searchsorted(cumulative_distance, next_distance)
        if next_index == 0:
            next_position = trajectory_enu[0]
        elif next_index == N:
            next_position = trajectory_enu[-1]
        else:
            t = (next_distance - cumulative_distance[next_index - 1]) / (cumulative_distance[next_index] - cumulative_distance[next_index - 1])
            next_position = trajectory_enu[next_index - 1] + t * (trajectory_enu[next_index] - trajectory_enu[next_index - 1])
            
        new_positions.append(next_position)
        new_cumulative_distance.append(next_distance)
        new_speeds.append(speeds[next_index])
        timestamps.append(timestamps[-1] + delta_t)
        
        current_distance = next_distance
        current_index = next_index
        
    new_positions = np.array(new_positions)
    new_cumulative_distance = np.array(new_cumulative_distance)
    new_speeds = np.array(new_speeds)
    timestamps = np.array(timestamps)

    # ---------- 朝向（四元数） enu ----------
    print("Computing smooth direction vectors...")
    directions = smooth_direction_vectors(new_positions, window_size=5)
    
    print("Computing smooth quaternions...")
    quaternions = compute_smooth_quaternions(directions)
    
    # ---------- 保存 pose.txt ----------
    pose = np.hstack((
        timestamps[:, None],
        new_positions,
        quaternions
    ))
    np.savetxt(output_file, pose, fmt="%.6f", delimiter=",",
              header="timestamp,tx,ty,tz,qx,qy,qz,qw", comments='')

    # ---------- 速度图 ----------
    plt.figure(figsize=(10, 4))
    plt.plot(new_cumulative_distance, new_speeds, label="Speed (m/s)")
    plt.xlabel("Cumulative Distance")
    plt.ylabel("Speed")
    plt.title("Velocity Profile")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("velocity_profile.png")
    plt.close()

if __name__ == "__main__":
  
    traj_path = "trajectory_3d.csv"
    output_pose_file = "pose.txt"
    
    # 读取 3D 轨迹数据
    traj = np.loadtxt(traj_path, delimiter=',', skiprows=1)
    traj[:, 1] = 60 - traj[:, 1]  # 把坐标中心换到左下（因为图片的0，0是左上），60是图片高度
    
    # 插值并平滑轨迹
    num_points = 4000
    smooth_traj = smooth_and_interpolate(traj, num_points=num_points, smooth=0)
    
    # 可视化插值后的轨迹，并每隔100个点标注编号
    # plot_trajectory_with_labels(smooth_traj, label_interval=20)
    
    # 给出分段平均速度，按照速度参考图给，单位是km/h
    speed_segments = [
        (0, 80, 0.0, 6.0),
        (80, 480, 6.0, 6.0),
        (480, 540, 6.0, 3.0),
        (540, 640, 3.0, 3.0),
        (640, 660, 3.0, 6.0),
        (660, 1120, 6.0, 48.0),
        (1120, 1240, 48.0, 45.0),
        (1240, 1340, 45.0, 30.0),
        (1340, 1880, 30.0, 20.0),
        (1880, 2040, 20.0, 30.0),
        (2040, 2140, 30.0, 20.0),
        (2140, 2360, 20.0, 25.0),
        (2360, 2480, 25.0, 30.0),
        (2480, 2980, 30.0, 35.0),
        (2980, 3260, 35.0, 20.0),
        (3260, 3460, 20.0, 15.0),
        (3460, 3800, 15.0, 6.0),
        (3800, 4000, 6.0, 0.0)
    ]
    
    generate_pose(smooth_traj, speed_segments, output_pose_file, 60)
  
  
