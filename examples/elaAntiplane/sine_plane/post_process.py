import processing.post as post
import processing.plot as plot
import processing.readwritedatafiles as readwritedatafiles

comp = 'StrainZX' # StrainZX VelocityZ

folder = "output/"
prefix = "Data_o3n50"

fname = folder+prefix+"_1.pkl"
solver = readwritedatafiles.read_data_file(fname)

# Unpack
mesh = solver.mesh
physics = solver.physics

plot.prepare_plot(close_all=False, linewidth=1.5)
### Line probe (y = 1) ###
# Parameters
xy1 = [1.,-0.1]; xy2 = [5.,-0.1]
# Initial condition
plot.plot_line_probe(mesh, physics, solver, comp, xy1=xy1, xy2=xy2,
		plot_numerical=False, plot_exact=False, plot_IC=True,
		create_new_figure=True, ylabel=None, vs_x=True, fmt="k-.",
		legend_label=None)
# Exact solution

for i_f in range(10):
	# Read data file
	fname = folder + prefix + f"_{i_f*2+1}.pkl"
	solver = readwritedatafiles.read_data_file(fname)

	# Unpack
	mesh = solver.mesh
	physics = solver.physics

	# Compute L2 error
	post.get_error(mesh, physics, solver, comp)

	''' Plot '''
	# # Density contour
	# plot.prepare_plot(linewidth=0.5)
	# plot.plot_solution(mesh, physics, solver, "VelocityZ", plot_numerical=True,
	# 		plot_exact=False, plot_IC=False, create_new_figure=True, fmt='bo',
	# 		legend_label="DG", include_mesh=True, regular_2D=True,
	# 		show_elem_IDs=False)
	# plot.save_figure(file_name='vz2D', file_type='pdf', crop_level=2)

	# DG solution
	plot.plot_line_probe(mesh, physics, solver, comp, xy1=xy1, xy2=xy2,
			num_pts=101,plot_numerical=True, plot_exact=False, plot_IC=False,
			create_new_figure=False, fmt="b-", legend_label=None, lw=i_f*0.1)


# Save figure
plot.save_figure(file_name='line_o3n50_epzx', file_type='pdf', crop_level=2)


# Density contour
plot.prepare_plot(linewidth=0.5)
plot.plot_solution(mesh, physics, solver, comp, plot_numerical=True,
		plot_exact=False, plot_IC=False, create_new_figure=True, fmt='bo',
		legend_label="DG", include_mesh=True, regular_2D=True,
		show_elem_IDs=False)
plot.save_figure(file_name='vz2D_o3n50_epzx', file_type='pdf', crop_level=2)
# plot.show_plot()
