# Define a threshold:
THRESHOLD = 0.075

# Get the max value along the y-axis:
y_max = max(pcd.points, key=lambda x: x[1])[1]

# Get the original points color to be updated:
pcd_colors = np.asarray(pcd.colors)

# Number of points:
n_points = pcd_colors.shape[0]

# update color:
for i in range(n_points):
    # if the current point is aground point:
    if pcd.points[i][1] >= y_max - THRESHOLD:
        pcd_colors[i] = GREEN  # color it green

pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

# Display:
o3d.visualization.draw_geometries([pcd, origin])