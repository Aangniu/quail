import numpy as np
from matplotlib import pyplot as plt

from scipy.interpolate import LinearNDInterpolator

import meshing.tools as mesh_tools

from processing.plot import get_sample_points
import copy as cpObj

class OutputPoint:
    def __init__(self, dt, xs):
        """
        Initializes a PointSource object.

        Parameters:
        - dt: float
            The time step size of the point source input.
        - xs: tuple or list
            The location of the point source (e.g., a tuple with coordinates).
        - data: np.ndarray
            A NumPy array containing the time series data of the point source.
        """
        self.ele_ID = -1  # Default value for element ID, can be set later
        self.dt = dt
        self.xs = xs
        self.pt_id = -1
        self.data = []

    def get_eId(self, eId):
        self.ele_ID = eId

    def __repr__(self):
        return f'Point {self.pt_id} at {self.xs}, with data {self.data}'

class PointOutput:
    def __init__(self, output_prefix):
        self.points = []
        self.num_points = 0
        self.output_prefix = output_prefix

    def add_point(self, point):
        point.pt_id = self.num_points
        self.points.append(point)
        self.num_points += 1
        print(point)

    def record_points_at_t(self, solver):
        mesh = solver.mesh
        physics = solver.physics
        basis = cpObj.deepcopy(solver.basis)

        pts = self.points

        ndims = mesh.ndims
        U = solver.state_coeffs
        order = solver.order

        # Get sample points in reference space
        xref = basis.equidistant_nodes(max([1, 3*order]))

        # Evaluate basis at reference-space points
        basis.get_basis_val_grads(xref, True, False, False, None)

        for pt in pts:
            xphys = mesh_tools.ref_to_phys(mesh, pt.ele_ID, xref)

            basis_val = basis.basis_val

            interpolator = LinearNDInterpolator(xphys, basis_val)
            basis_values_at_sou = interpolator(pt.xs)

            # print(basis_values_at_sou)

            U_ele = solver.state_coeffs[pt.ele_ID,:,:]
            # print(U_ele.shape)
            vars_U = np.einsum('ij,jk->ik', basis_values_at_sou, U_ele)
            # var_out = physics.compute_variable("VelocityZ", vars_U)
            var_out = vars_U[0,:]
            # print("vars_U: ", vars_U)

            pt.data.append(var_out)
        # print(pts[0])

    def write_points(self):
        # with open(self.output_path, 'w') as f:
        for point in self.points:
            # print("recorded point: ", point)
            # Convert list of arrays into a single 2D array for saving
            combined_data = np.vstack(point.data)
            # print(combined_data)
            np.savetxt(self.output_prefix + f"_{point.pt_id}" + ".dat"\
                      , combined_data, fmt='%.8e', delimiter=' ')

            plt.figure()
            plt.plot(point.data)
            plt.show()

            # f.write(f'{point.x},{point.y}\n')

    def get_cell_element_id(self, mesh):
        for elem_ID in range(mesh.num_elems):
            cell_phys_nodes = mesh.elements[elem_ID].node_coords
            x_sources = np.array([ps.xs for ps in self.points])

			# TODO: when more than one source lie in a cell
            p_id = self.cell_contains(cell_phys_nodes, x_sources)
            # print(p_id)
            if p_id:
                # print(p_id)
                self.points[p_id[0]].get_eId(elem_ID)
        # make sure the sources are all in the mesh
        for ps in self.points:
            if ps.ele_ID == -1:
                raise ValueError(f"Point source {ps}, with coordinates {ps.xs}, "
								 "is not in the mesh.")

    def cell_contains(self, cell_phys_nodes, x_sources):
        '''
        This function returns whether x_source is inside the triangle
        with cell_phys_nodes as vertices.

        Inputs:
        -------
            x_sources: point source locations
            cell_phys_nodes: physical coords of the cell vertices
            basis: basis object

        Outputs:
        -------
            p_id: point_source id, [] is empty if not in cell
        '''

        def signs(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - \
                (p2[0] - p3[0]) * (p1[1] - p3[1])

        p_id = []
        for i in range(x_sources.shape[0]):
            # Check if x_sources[i] is inside the triangle
            d1 = signs(x_sources[i], cell_phys_nodes[0,:], cell_phys_nodes[1,:])
            d2 = signs(x_sources[i], cell_phys_nodes[1,:], cell_phys_nodes[2,:])
            d3 = signs(x_sources[i], cell_phys_nodes[2,:], cell_phys_nodes[0,:])

            has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
            has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

            if not (has_neg and has_pos):
                p_id.append(i)
                # p_id = i

        return p_id