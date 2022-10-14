points = np.asarray(pcd.points)
np.random.shuffle(points)
u_pcd= o3d.geometry.PointCloud()
u_pcd.points= o3d.utility.Vector3dVector(points)