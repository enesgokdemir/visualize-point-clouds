"""
  Convert from rgb camera coordinate system
  to rgb image coordinate system:
"""
j_rgb = int((x_RGB * FX_RGB) / z_RGB + CX_RGB + width / 2)
i_rgb = int((y_RGB * FY_RGB) / z_RGB + CY_RGB)