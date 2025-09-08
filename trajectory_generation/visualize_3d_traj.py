import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
import open3d as o3d

def plot_csv_trajectory(csv_path):
    print(f"读取 CSV: {csv_path}")
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    z = -z  # 可选反转 Z

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, 'b-', linewidth=2)
    ax.set_title("3D 轨迹可视化（CSV）")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.tight_layout()
    plt.show()

def plot_obj_trajectory_open3d(obj_path):
    print(f"读取 OBJ: {obj_path}")
    mesh = o3d.io.read_triangle_mesh(obj_path)

    if len(mesh.vertices) == 0:
        # 若为线条OBJ，则尝试手动解析为 LineSet
        vertices = []
        lines = []
        with open(obj_path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith('l '):
                    idx = list(map(lambda s: int(s)-1, line.strip().split()[1:]))  # OBJ索引从1开始
                    if len(idx) == 2:
                        lines.append(idx)

        if vertices and lines:
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(vertices),
                lines=o3d.utility.Vector2iVector(lines)
            )
            o3d.visualization.draw_geometries([line_set], window_name='3D轨迹 (OBJ)', width=800, height=600)
        else:
            print("OBJ 文件中没有可识别的线条或顶点。")
    else:
        print("这是一个三角网格，不是轨迹线。将以 mesh 显示。")
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh])

def main():
    parser = argparse.ArgumentParser(description="可视化 3D 轨迹（CSV 或 OBJ）")
    parser.add_argument('--file', type=str, required=True, help='输入文件路径（.csv 或 .obj）')
    args = parser.parse_args()

    ext = os.path.splitext(args.file)[-1].lower()
    if ext == ".csv":
        plot_csv_trajectory(args.file)
    elif ext == ".obj":
        plot_obj_trajectory_open3d(args.file)
    else:
        print("错误：仅支持 .csv 或 .obj 文件")

if __name__ == "__main__":
    main()