# Ground_detection:
THRESHOLD = 0.075 * 1000  # Define a threshold
y_max = max(organized_pcd.reshape((height * width, 3)), key=lambda x: x[1])[
    1]  # Get the max value along the y-axis

# Set the ground pixels to green:
for i in range(height):
    for j in range(width):
        if organized_pcd[i][j][1] >= y_max - THRESHOLD:
            depth_grayscale[i][j] = [0, 255, 0]  # Update the depth image

# Display depth_grayscale:
plt.imshow(depth_grayscale)
plt.show()