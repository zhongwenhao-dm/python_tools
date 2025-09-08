import open3d as o3d
import numpy as np
from scipy.interpolate import splprep, splev

def fit_curve_through_points(points, samples=500, smooth=0.0):
    """用三维样条拟合点，并输出采样曲线"""
    points = np.asarray(points)
    if points.shape[0] < 2:
        raise ValueError("至少需要两个点来拟合曲线")

    # 用参数化样条拟合
    tck, u = splprep([points[:, 0], points[:, 1], points[:, 2]], s=smooth, per=True)
    u_fine = np.linspace(0, 1, samples)
    x, y, z = splev(u_fine, tck)
    curve = np.vstack([x, y, z]).T
    return curve

if __name__ == "__main__":
    # 加载点云
    pcd = o3d.io.read_point_cloud("./data/fitted_curve_coarse.ply")

    # 打开交互式选择器
    print("请在窗口中选择点，按 [Shift + 左键] 选点，按 Q 或 ESC 退出。")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

    picked_ids = vis.get_picked_points()
    pts = np.asarray(pcd.points)[picked_ids]
    print("\n你选中了以下点：")
    for i, p in enumerate(pts):
        print(f"{i}: {p}")

    # ==== 删除功能 ====
    delete_str = input("\n请输入要删除的点的序号(用空格分隔)，或直接回车跳过: ")
    if delete_str.strip():
        delete_ids = list(map(int, delete_str.split()))
        mask = np.ones(len(pts), dtype=bool)
        mask[delete_ids] = False
        pts = pts[mask]
        print(f"删除后剩余 {len(pts)} 个点")

    if len(pts) >= 2:
        # 按选点顺序拟合曲线
        curve = fit_curve_through_points(pts, samples=1000, smooth=0.01)

        # 可视化
        curve_pcd = o3d.geometry.PointCloud()
        curve_pcd.points = o3d.utility.Vector3dVector(curve)
        curve_pcd.paint_uniform_color([1, 0, 0])  # 红色曲线

        picked_pcd = o3d.geometry.PointCloud()
        picked_pcd.points = o3d.utility.Vector3dVector(pts)
        picked_pcd.paint_uniform_color([0, 1, 0])  # 绿色选点

        o3d.visualization.draw_geometries([pcd, picked_pcd, curve_pcd])

        # 保存
        o3d.io.write_point_cloud("./data/fitted_curve.ply", curve_pcd)
        np.savetxt("./data/fitted_curve.txt", curve, fmt="%.6f")
        print("拟合曲线已保存到 ./data/fitted_curve.ply 和 fitted_curve.txt")
    else:
        print("没有足够的点进行拟合！")
