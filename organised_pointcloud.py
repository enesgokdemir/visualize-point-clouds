# get depth image resolution:
height, width = depth_image.shape
# compute indices and reshape it to have the same shape as the depth image:
jj = np.tile(range(width), height).reshape((height, width))
ii = np.repeat(range(height), width).reshape((height, width))
# Compute constants:
xx = (jj - CX_DEPTH) / FX_DEPTH
yy = (ii - CY_DEPTH) / FY_DEPTH
# compute organised point cloud:
organized_pcd = np.dstack((xx * depth_image, yy * depth_image, depth_image))