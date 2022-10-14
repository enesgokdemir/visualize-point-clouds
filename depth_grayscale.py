depth_instensity = np.array(256 * depth_image / 0x0fff,
                            dtype=np.uint8)
iio.imwrite('output/grayscale.png', depth_instensity)