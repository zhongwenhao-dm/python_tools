import open3d as o3d


pcd_file1 = './data/cut1.ply'
pcd_file2 = './data/fitted_curve.ply'

pcd1 = o3d.io.read_point_cloud(pcd_file1)
pcd2 = o3d.io.read_point_cloud(pcd_file2)

o3d.visualization.draw_geometries([pcd1, pcd2])