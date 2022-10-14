import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt

# Camera parameters:
FX_DEPTH = 5.8262448167737955e+02
FY_DEPTH = 5.8269103270988637e+02
CX_DEPTH = 3.1304475870804731e+02
CY_DEPTH = 2.3844389626620386e+02

# Read depth image:
depth_image = iio.imread('../data/depth_2.png')
# Compute the grayscale image:
depth_grayscale = np.array(256 * depth_image / 0x0fff, dtype=np.uint8)
# Convert a grayscale image to a 3-channel image:
depth_grayscale = np.stack((depth_grayscale,) * 3, axis=-1)