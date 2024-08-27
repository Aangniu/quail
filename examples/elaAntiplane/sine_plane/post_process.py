import processing.post as post
import processing.plot as plot
import processing.readwritedatafiles as readwritedatafiles

fname = "Data_1.pkl"
solver = readwritedatafiles.read_data_file(fname)

# Unpack
mesh = solver.mesh
physics = solver.physics

plot.prepare_plot(close_all=False, linewidth=1.5)
### Line probe (y = 1) ###
# Parameters
xy1 = [-1.,0.001]; xy2 = [1.,0.001]
# Initial condition
plot.plot_line_probe(mesh, physics, solver, "VelocityZ", xy1=xy1, xy2=xy2,
		plot_numerical=False, plot_exact=False, plot_IC=True,
		create_new_figure=True, ylabel=None, vs_x=True, fmt="k-.",
		legend_label=None)
# Exact solution

for i_f in range(20):
	# Read data file
	fname = f"Data_{i_f+1}.pkl"
	solver = readwritedatafiles.read_data_file(fname)

	# Unpack
	mesh = solver.mesh
	physics = solver.physics

	# Compute L2 error
	post.get_error(mesh, physics, solver, "VelocityZ")

	''' Plot '''
	# # Density contour
	# plot.prepare_plot(linewidth=0.5)
	# plot.plot_solution(mesh, physics, solver, "VelocityZ", plot_numerical=True,
	# 		plot_exact=False, plot_IC=False, create_new_figure=True, fmt='bo',
	# 		legend_label="DG", include_mesh=True, regular_2D=True,
	# 		show_elem_IDs=False)
	# plot.save_figure(file_name='vz2D', file_type='pdf', crop_level=2)

	# DG solution
	plot.plot_line_probe(mesh, physics, solver, "VelocityZ", xy1=xy1, xy2=xy2,
			num_pts=101,plot_numerical=True, plot_exact=False, plot_IC=False,
			create_new_figure=False, fmt="b-", legend_label=None)


# Save figure
plot.save_figure(file_name='line', file_type='pdf', crop_level=2)

# plot.show_plot()
