# compute indices:
jj = np.tile(range(width), height)
ii = np.repeat(range(height), width)
# Compute constants:
xx = (jj - CX_DEPTH) / FX_DEPTH
yy = (ii - CY_DEPTH) / FY_DEPTH
# transform depth image to vector of z:
length = height * width
z = depth_image.reshape(height * width)
# compute point cloud
pcd = np.dstack((xx * z, yy * z, z)).reshape((length, 3))