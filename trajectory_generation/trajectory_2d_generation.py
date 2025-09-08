import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splprep, splev

import os

# 屏蔽 OpenCV 的 Qt 插件路径
os.environ.pop("QT_PLUGIN_PATH", None)

# 指定系统 Qt 插件路径
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms"

points = []

def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        x, y = event.xdata, event.ydata
        points.append([x, y])
        plt.plot(x, y, 'ro')
        plt.draw()

def smooth_curve(points, num_points=200, smoothness=0.0):
    points = np.array(points)
    x, y = points[:, 0], points[:, 1]
    
    # 拟合 B-spline 曲线
    tck, u = splprep([x, y], s=smoothness)
    u_fine = np.linspace(0, 1, num_points)
    x_fine, y_fine = splev(u_fine, tck)
    smoothed = np.vstack((x_fine, y_fine)).T
    return smoothed

def main(image_path, output_csv):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots()
    ax.imshow(img_rgb)
    ax.set_title("点击轨迹点（建议 10+ 个），关闭窗口后平滑并保存")

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    if len(points) >= 4:
        # 平滑插值轨迹
        smoothed_path = smooth_curve(points, num_points=300, smoothness=5.0)

        # 保存平滑后的轨迹
        np.savetxt(output_csv, smoothed_path, delimiter=',', header='x,y', comments='')
        print(f"平滑 2D 轨迹已保存到 {output_csv}")

        # 可视化对比原始与平滑
        plt.figure(figsize=(10, 6))
        plt.imshow(img_rgb)
        original = np.array(points)
        plt.plot(original[:, 0], original[:, 1], 'ro-', label='原始点')
        plt.plot(smoothed_path[:, 0], smoothed_path[:, 1], 'b-', linewidth=2, label='平滑曲线')
        plt.legend()
        plt.title("2D轨迹平滑效果")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("trajectory_2d_smooth_preview.png")
        plt.show()

    else:
        print("点太少，无法进行平滑。请至少点击 4 个点。")

if __name__ == "__main__":
    main("track_line_projection.png", "trajectory_2d.csv")