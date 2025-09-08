import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import cv2
from mpl_toolkits.mplot3d import Axes3D  # 注册 3D 支持

import os

# 屏蔽 OpenCV 的 Qt 插件路径
os.environ.pop("QT_PLUGIN_PATH", None)

# 指定系统 Qt 插件路径
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms"

def cumulative_distance(pts):
    d = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
    return np.insert(np.cumsum(d), 0, 0)

def onclick_z(event, trajectory_2d, s_vals, clicked_heights):
    if event.xdata is not None and event.ydata is not None:
        x, y = event.xdata, event.ydata
        z = float(input(f"为点 ({x:.1f}, {y:.1f}) 输入高度 z（单位米）："))
        dist = np.linalg.norm(trajectory_2d - np.array([x, y]), axis=1)
        idx = np.argmin(dist)
        s = s_vals[idx]
        clicked_heights.append((s, z))
        plt.plot(x, y, 'go')
        plt.draw()

def show_3d_plot(traj_3d):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(traj_3d[:, 0], traj_3d[:, 1], traj_3d[:, 2], 'b-')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (高度)')
    ax.set_title("3D 轨迹可视化")
    plt.tight_layout()
    plt.show()

def save_obj_file(trajectory_3d, filename="trajectory.obj"):
    with open(filename, 'w') as f:
        for x, y, z in trajectory_3d:
            f.write(f"v {x:.4f} {y:.4f} {z:.4f}\n")
        for i in range(1, len(trajectory_3d)):
            f.write(f"l {i} {i+1}\n")
    print(f"已保存为 .obj 文件：{filename}")

def main(image_path, trajectory_csv, output_csv, scale_width_meters=67):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    trajectory_2d = np.loadtxt(trajectory_csv, delimiter=',', skiprows=1)
    s_vals = cumulative_distance(trajectory_2d)
    clicked_heights = []

    # 2D 轨迹 + 标注 Z 点
    fig, ax = plt.subplots()
    ax.imshow(img_rgb)
    ax.plot(trajectory_2d[:, 0], trajectory_2d[:, 1], 'r-', label="轨迹")
    ax.set_title("点击轨迹上点以标注高度 z，关闭窗口后生成 3D")
    fig.canvas.mpl_connect('button_press_event', lambda event: onclick_z(event, trajectory_2d, s_vals, clicked_heights))
    plt.show()

    if len(clicked_heights) >= 2:
        clicked_heights.sort()
        s_clicked, z_clicked = zip(*clicked_heights)
        f_z = interp1d(s_clicked, z_clicked, kind='cubic', fill_value='extrapolate')
        z_vals = f_z(s_vals)
        
        # 把像素值换算为真实的比例
        _, w = img.shape[:2]
        ratio = scale_width_meters / w  # 假设图像宽度对应 67 米
        trajectory_2d = trajectory_2d * ratio
        
        # 合并成3d
        trajectory_3d = np.hstack((trajectory_2d, z_vals[:, None]))
        
        # 保存为 CSV
        np.savetxt(output_csv, trajectory_3d, delimiter=',', header='x,y,z', comments='')
        print(f"已保存 3D 轨迹至 {output_csv}")

        # 显示 3D 图
        show_3d_plot(trajectory_3d)

        # 保存为 OBJ 文件
        save_obj_file(trajectory_3d, "trajectory.obj")

    else:
        print("高度点太少，未生成 3D 轨迹")

if __name__ == "__main__":
    main("track_line_projection.png", "trajectory_2d.csv", "trajectory_3d.csv")