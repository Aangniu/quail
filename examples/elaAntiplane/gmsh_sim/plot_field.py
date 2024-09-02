import processing.post as post
import processing.plot as plot
import processing.readwritedatafiles as readwritedatafiles

comp = 'StrainZX' # StrainZX VelocityZ

folder = "output/"

fname = folder+"Data_Agmsh_10.pkl"
solver = readwritedatafiles.read_data_file(fname)

# Unpack
mesh = solver.mesh
physics = solver.physics

# Density contour
plot.prepare_plot(linewidth=0.5)
plot.plot_solution(mesh, physics, solver, comp, plot_numerical=True,
		plot_exact=False, plot_IC=False, create_new_figure=True, fmt='bo',
		legend_label="DG", include_mesh=True, regular_2D=True,
		show_elem_IDs=False)
plot.save_figure(file_name='Agmsh_epzx', file_type='pdf', crop_level=2)
# plot.show_plot()

comp = 'VelocityZ' # StrainZX VelocityZ
# Density contour
plot.prepare_plot(linewidth=0.5)
plot.plot_solution(mesh, physics, solver, comp, plot_numerical=True,
		plot_exact=False, plot_IC=False, create_new_figure=True, fmt='bo',
		legend_label="DG", include_mesh=True, regular_2D=True,
		show_elem_IDs=False)
plot.save_figure(file_name='Agmsh_vz', file_type='pdf', crop_level=2)
# plot.show_plot()