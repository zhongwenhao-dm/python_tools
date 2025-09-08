import open3d as o3d
import numpy as np
from PIL import Image, ImageDraw

def transform_points(points, scale=1.0, rotation_deg=0.0, translation=(0, 0)):
    # scale -> rotate -> translate
    theta = np.deg2rad(rotation_deg)
    rot_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    return (points @ rot_matrix.T) * scale + np.array(translation)

# === 1. 加载轨道模型 ===
mesh = o3d.io.read_triangle_mesh("guidao.obj")
mesh.compute_adjacency_list()  # 预处理
vertices = np.asarray(mesh.vertices)

# === 2. 投影到（顶视角） ===
xy = vertices[:, [0, 2]]  # 忽略 y

# === 3. 映射到图像空间 ===
img = Image.open("image.jpg").convert("RGB")
w, h = img.size

# 归一化 -> 图像坐标
min_xy = xy.min(axis=0)
max_xy = xy.max(axis=0)
norm_xy = (xy - min_xy) / (max_xy - min_xy)
img_xy = (norm_xy * [w - 1, h - 1]).astype(int)
# img_xy[:, 1] = h - img_xy[:, 1]  # 翻转Y坐标

# 应用仿射变换
scale = 0.95
rotation_deg = 0
translation = [25, 10]
img_xy = transform_points(img_xy, scale, rotation_deg, translation)

# === 4. 从网格中提取边线连接 ===
lines = np.asarray(mesh.triangles)  # 每个三角面连接的3个点
edges = set()
for tri in lines:
    edges.add(tuple(sorted((tri[0], tri[1]))))
    edges.add(tuple(sorted((tri[1], tri[2]))))
    edges.add(tuple(sorted((tri[2], tri[0]))))

# === 5. 画线 ===
draw = ImageDraw.Draw(img)
for idx0, idx1 in edges:
    x0, y0 = img_xy[idx0]
    x1, y1 = img_xy[idx1]
    draw.line([(x0, y0), (x1, y1)], fill="green", width=1)

# === 6. 保存或展示结果 ===
img.save("track_line_projection.png")
img.show()