# Get max and min points of each axis x, y and z:
x_max = max(pcd.points, key=lambda x: x[0])
y_max = max(pcd.points, key=lambda x: x[1])
z_max = max(pcd.points, key=lambda x: x[2])
x_min = min(pcd.points, key=lambda x: x[0])
y_min = min(pcd.points, key=lambda x: x[1])
z_min = min(pcd.points, key=lambda x: x[2])