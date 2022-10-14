# get depth resolution:
height, width = depth_im.shape
length = height * width
# compute indices:
jj = np.tile(range(width), height)
ii = np.repeat(range(height), width)
# rechape depth image
z = depth_im.reshape(length)
# compute pcd:
pcd = np.dstack([(ii - CX_DEPTH) * z / FX_DEPTH,
                 (jj - CY_DEPTH) * z / FY_DEPTH,
                 z]).reshape((length, 3))