# Display depth and grayscale image:
fig, axs = plt.subplots(1, 2)
axs[0].imshow(depth_image, cmap="gray")
axs[0].set_title('Depth image')
axs[1].imshow(depth_grayscale, cmap="gray")
axs[1].set_title('Depth grayscale image')
plt.show()