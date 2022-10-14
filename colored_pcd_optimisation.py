# compute indices:
jj = np.tile(range(width), height)
ii = np.repeat(range(height), width)

# Compute constants:
xx = (jj - CX_DEPTH) / FX_DEPTH
yy = (ii - CY_DEPTH) / FY_DEPTH

# transform depth image to vector of z:
length = height * width
z = depth_image.reshape(length)

# compute point cloud
pcd = np.dstack((xx * z, yy * z, z)).reshape((length, 3))
cam_RGB = np.apply_along_axis(np.linalg.inv(R).dot, 1, pcd) - np.linalg.inv(R).dot(T)
xx_rgb = ((cam_RGB[:, 0] * FX_RGB) / cam_RGB[:, 2] + CX_RGB + width / 2).astype(int).clip(0, width - 1)
yy_rgb = ((cam_RGB[:, 1] * FY_RGB) / cam_RGB[:, 2] + CY_RGB).astype(int).clip(0, height - 1)
colors = rgb_image[yy_rgb, xx_rgb]