# Read point cloud:
pcd = o3d.io.read_point_cloud("data/depth_2_pcd.ply")
# Create a 3D coordinate system:
origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
# geometries to draw:
geometries = [pcd, origin]
# Visualize:
o3d.visualization.draw_geometries(geometries)