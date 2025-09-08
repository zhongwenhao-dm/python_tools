import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.interpolate import splprep, splev
import matplotlib.cm as cm

def get_color_pointcloud(pcd, target_color, threshold):
    # Open3D 的颜色是 [0,1] 范围
    colors = np.asarray(pcd.colors)
    points = np.asarray(pcd.points)
    
    distances = np.linalg.norm(colors - target_color, axis=1)
    mask = distances < threshold
    
    selected_points = points[mask]
    selected_colors = colors[mask]
    
    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(selected_points)
    new_pcd.colors = o3d.utility.Vector3dVector(selected_colors)
    
    print(f"筛选到 {len(selected_points)} 个接近{target_color} 的点")
  
    return new_pcd


def extract_center_line(pcd,
                        n_bins=180,          # 极角分桶数量（越大越细）
                        min_pts=30,          # 每个桶最少点数
                        use_kmeans=True,     # 是否用KMeans把两轨分成两簇
                        q_inner=0.2, q_outer=0.8,  # 不用KMeans时的内外径分位数
                        smooth_ratio=0.01,   # 样条平滑强度比例（对点数自适应）
                        samples=1500):       # 输出中心线采样点数
    """
    从闭环双轨点云中提取中心线（适合近似环形/椭圆，有高低起伏的情况）
    思路：PCA主平面 → 极角分桶 → 每桶按半径分两轨 → 取两轨3D均值的中点 → 闭合样条拟合
    """
    pts = np.asarray(pcd.points)
    if len(pts) < 100:
        raise ValueError("点太少，无法提取中心线")

    # 1) PCA 主平面（用SVD快速做）
    C = pts.mean(axis=0)
    U, S, Vt = np.linalg.svd(pts - C, full_matrices=False)
    a, b, n = Vt[0], Vt[1], Vt[2]   # 平面两基和法向

    # 2) 投影到主平面，转极坐标（θ 用于排序，r 用于两轨分离）
    P = pts - C
    u = P @ a
    v = P @ b
    theta = np.arctan2(v, u) % (2*np.pi)
    r = np.sqrt(u*u + v*v)

    # 3) 极角分桶，逐桶求“中心点”
    bins = np.linspace(0, 2*np.pi, n_bins+1)
    centers3d, angles = [], []
    bucket_pcds = []   # 保存每个桶的点云（不同颜色）
    colormap = cm.get_cmap("hsv", n_bins)  # 彩虹色循环
    for i in range(n_bins):
        m = (theta >= bins[i]) & (theta < bins[i+1])
        idx = np.where(m)[0]
        if idx.size < min_pts:
            continue

        ri = r[idx]
        
        # ====== 桶内点云着色 ======
        bucket = o3d.geometry.PointCloud()
        bucket.points = o3d.utility.Vector3dVector(pts[idx])
        color = colormap(i)[:3]  # 取 (r,g,b)
        bucket.paint_uniform_color(color)
        bucket_pcds.append(bucket)
        
        # 用半径在1D上分两簇：内轨/外轨
        if use_kmeans:
            km = KMeans(n_clusters=2, n_init=5, random_state=0).fit(ri.reshape(-1,1))
            labels = km.labels_
            counts = np.bincount(labels)
            # 若分簇很不稳定，退化到分位数法
            if counts.min() < max(5, int(0.1*idx.size)):
                use_local_kmeans = False
            else:
                use_local_kmeans = True
        else:
            use_local_kmeans = False

        if use_local_kmeans:
            c1 = pts[idx[labels==0]].mean(axis=0)
            c2 = pts[idx[labels==1]].mean(axis=0)
            mid = 0.5*(c1 + c2)
        else:
            # 分位数法：用 r 的内/外分位对应的邻近点集合求均值
            r_in = np.quantile(ri, q_inner)
            r_out = np.quantile(ri, q_outer)
            in_mask = ri <= r_in
            out_mask = ri >= r_out
            if in_mask.sum() < 3 or out_mask.sum() < 3:
                continue
            c1 = pts[idx[in_mask]].mean(axis=0)
            c2 = pts[idx[out_mask]].mean(axis=0)
            mid = 0.5*(c1 + c2)

        centers3d.append(mid)
        angles.append(0.5*(bins[i] + bins[i+1]))

    centers3d = np.asarray(centers3d)
    angles = np.asarray(angles)
    if centers3d.shape[0] < 20:
        raise RuntimeError("有效中心点太少，尝试降低 min_pts 或增大 n_bins")

    # 4) 按角度排序，并用“闭合样条”拟合（per=True）
    order = np.argsort(angles)
    centers3d = centers3d[order]
    # 平滑强度：对点数自适应，越大越平滑
    s = smooth_ratio * centers3d.shape[0]
    tck, _ = splprep([centers3d[:,0], centers3d[:,1], centers3d[:,2]],
                     s=s, k=3, per=True)
    u_f = np.linspace(0, 1, samples)
    x, y, z = splev(u_f, tck)
    curve = np.vstack([x, y, z]).T

    # 5) 输出为点云（蓝色为连续中心线，红色为离散中点）
    line_pcd = o3d.geometry.PointCloud()
    line_pcd.points = o3d.utility.Vector3dVector(curve)
    line_pcd.paint_uniform_color([0, 0, 1])

    mids_pcd = o3d.geometry.PointCloud()
    mids_pcd.points = o3d.utility.Vector3dVector(centers3d)
    mids_pcd.paint_uniform_color([1, 0, 0])

    print(f"Center points used: {len(centers3d)}  |  Output samples: {samples}")
    return line_pcd, mids_pcd, bucket_pcds


if __name__ == '__main__':
    input_ply_file = './data/cut1.ply'
    output_ply_file = './data/orange_points.ply'
    centerline_file = './data/centerline.ply'

    pcd = o3d.io.read_point_cloud(input_ply_file)

    # 先提取轨道点
    orange = np.array([1.0, 0.65, 0.0])
    threshold = 0.6  # 容差，越小筛选越严格
    new_pcd = get_color_pointcloud(pcd, orange, threshold)
    
    line_pcd, mids_pcd, bucket_pcds = extract_center_line(new_pcd,
                                            n_bins=240,
                                            min_pts=10,
                                            use_kmeans=True,
                                            smooth_ratio=0.005,
                                            samples=2000)
    o3d.io.write_point_cloud('./data/centerline.ply', line_pcd)
    # o3d.visualization.draw_geometries([*bucket_pcds, mids_pcd])
    o3d.visualization.draw_geometries([new_pcd, mids_pcd, line_pcd, *bucket_pcds])
    
    
    o3d.io.write_point_cloud(output_ply_file, new_pcd+mids_pcd)