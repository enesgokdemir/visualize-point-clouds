"""
  Convert the point from depth sensor 3D coordinate system
  to rgb camera coordinate system:
"""
[x_RGB, y_RGB, z_RGB] = np.linalg.inv(R).dot([x, y, z]) - np.linalg.inv(R).dot(T)