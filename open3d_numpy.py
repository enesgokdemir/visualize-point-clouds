# Read the bunny point cloud file:
pcd_o3d = o3d.io.read_point_cloud("../data/bunny_pcd.ply")

# Convert the open3d object to numpy:
pcd_np = np.asarray(pcd_o3d.points)

# Display using matplotlib:
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.scatter3D(pcd_np[:, 0], pcd_np[:, 2], pcd_np[:, 1])
# label the axes
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Bunny Point Cloud")
# display:
plt.show()