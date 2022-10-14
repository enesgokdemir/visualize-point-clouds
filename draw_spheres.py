# Colors:
RED = [1., 0., 0.]
GREEN = [0., 1., 0.]
BLUE = [0., 0., 1.]
YELLOW = [1., 1., 0.]
MAGENTA = [1., 0., 1.]
CYAN = [0., 1., 1.]

positions = [x_max, y_max, z_max, x_min, y_min, z_min]
colors = [RED, GREEN, BLUE, MAGENTA, YELLOW, CYAN]
for i in range(len(positions)):
   # Create a sphere mesh:
   sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
   # move to the point position:
   sphere.translate(np.asarray(positions[i]))
   # add color:
   sphere.paint_uniform_color(np.asarray(colors[i]))
   # compute normals for vertices or faces:
   sphere.compute_vertex_normals()
   # add to geometry list to display later:
   geometries.append(sphere)

# Display:
o3d.visualization.draw_geometries(geometries)