# ------------------------------------------------------------------------ #
#
#       quail: A lightweight discontinuous Galerkin code for
#              teaching and prototyping
#		<https://github.com/IhmeGroup/quail>
#
#		Copyright (C) 2020-2021
#
#       This program is distributed under the terms of the GNU
#		General Public License v3.0. You should have received a copy
#       of the GNU General Public License along with this program.
#		If not, see <https://www.gnu.org/licenses/>.
#
# ------------------------------------------------------------------------ #

# ------------------------------------------------------------------------ #
#
#       File : src/physics/elaAntiplane/elaAntiplane.py
#
#       Contains class definitions for 2D anti-plane wave equations.
#
# ------------------------------------------------------------------------ #
from enum import Enum
import numpy as np

from scipy.interpolate import LinearNDInterpolator
import copy as cpObj

import numerics.basis.tools as basis_tools

import errors
import general
import meshing.tools as mesh_tools

import physics.base.base as base
import physics.base.functions as base_fcns
from physics.base.functions import BCType as base_BC_type
from physics.base.functions import ConvNumFluxType as base_conv_num_flux_type
from physics.base.functions import FcnType as base_fcn_type

import physics.elaAntiplane.functions as antiWave_fcns
from physics.elaAntiplane.functions import BCType as antiWave_BC_type

from physics.elaAntiplane.functions import FcnType as antiWave_fcn_type
from physics.elaAntiplane.functions import SourceType as \
	antiWave_source_type

# from physics.euler.functions import ConvNumFluxType as \
# 		euler_conv_num_flux_type

class PointSource:
    def __init__(self, dt, xs, data):
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
        self.data = np.array(data)  # Ensure data is stored as a NumPy array

    def __repr__(self):
        return (f"PointSource(ele_ID={self.ele_ID}, dt={self.dt}, xs={self.xs}, "
                f"data=Array of length {len(self.data)})")

    def set_eleID(self, ele_ID):
        """
        Sets the element ID for the point source.

        Parameters:
        - ele_ID: int
            The element ID to set for the point source.
        """
        self.ele_ID = ele_ID

def cell_contains(cell_phys_nodes, x_sources):
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
		p_id: point_source id, -1 if not in cell
	'''

	def signs(p1, p2, p3):
		return (p1[0] - p3[0]) * (p2[1] - p3[1]) - \
			(p2[0] - p3[0]) * (p1[1] - p3[1])

	p_id = -1
	for i in range(x_sources.shape[0]):
		# Check if x_sources[i] is inside the triangle
		d1 = signs(x_sources[i], cell_phys_nodes[0,:], cell_phys_nodes[1,:])
		d2 = signs(x_sources[i], cell_phys_nodes[1,:], cell_phys_nodes[2,:])
		d3 = signs(x_sources[i], cell_phys_nodes[2,:], cell_phys_nodes[0,:])

		has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
		has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

		if not (has_neg and has_pos):
			p_id = i

	return p_id

class AntiplaneWave(base.PhysicsBase):
	'''
	This class corresponds to the compressible Euler equations for a
	calorically perfect gas. It inherits attributes and methods from the
	PhysicsBase class. See PhysicsBase for detailed comments of attributes
	and methods. This class should not be instantiated directly. Instead,
	the 1D and 2D variants, which inherit from this class (see below),
	should be instantiated.

	Additional methods and attributes are commented below.

	Attributes:
	-----------
	mu: float
		elastic shear modulus
	rho: float
		solid density
	point_sources: numpy array of the PointSource structure
	    The PointSource structure include:
			ele_ID: element ID,
			dt: time step size of the point source input,
			xs: location of the point source,
			data: time series data of the point source
	'''
	PHYSICS_TYPE = general.PhysicsType.ElaAntiplain

	def __init__(self):
		super().__init__()
		self.mu = 0.
		self.rho = 0.
		self.point_sources = []

	def set_maps(self):
		super().set_maps()

		self.BC_map.update({
			base_BC_type.StateAll : base_fcns.StateAll,
			base_BC_type.Extrapolate : base_fcns.Extrapolate,
		})

	def set_physical_params(self, ShearModulus=1e9, Density=1e3):
		'''
		This method sets physical parameters.

		Inputs:
		-------
			ShearModulus: solid shear modulus
			Density: solid density

		Outputs:
		--------
			self: physical parameters set
		'''
		self.mu = ShearModulus
		self.rho = Density

	def include_point_sources(self, mesh, physics_type):
		'''
		This method sets physical parameters.

		Inputs:
		-------
			ShearModulus: solid shear modulus
			Density: solid density

		Outputs:
		--------
			self: physical parameters set
		'''
		source_times = np.linspace(0,10,1001)
		# generate a source_data that formr gaussian function in time
		source_data = 1.0*np.exp(-(source_times - 2.0)**2/1.0**2)
		self.point_sources.append(PointSource(0.01, [-0.1001, -0.1001], source_data))

		for elem_ID in range(mesh.num_elems):
			cell_phys_nodes = mesh.elements[elem_ID].node_coords
			x_sources = np.array([ps.xs for ps in self.point_sources])

			# TODO: when more than one source lie in a cell
			p_id = cell_contains(cell_phys_nodes, x_sources)
			if p_id != -1:
				self.point_sources[p_id].set_eleID(elem_ID)
		# make sure the sources are all in the mesh
		for ps in self.point_sources:
			if ps.ele_ID == -1:
				raise ValueError(f"Point source {ps}, with coordinates {ps.xs}, "
								 "is not in the mesh.")

	class AdditionalVariables(Enum):
		# StateVariable = "\\Psi"
		WaveSpeed = "cs"
		MaxWaveSpeed = "\\lambda"
		# SlipRate = "|V|"
		# ZDisplacement = "D"

	def compute_additional_variable(self, var_name, Uq, flag_non_physical):
		''' Extract state variables '''
		sepzx = self.get_state_slice("StrainZX")
		sepyz = self.get_state_slice("StrainYZ")
		svz = self.get_state_slice("VelocityZ")
		epzx = Uq[:, :, sepzx]
		epyz = Uq[:, :, sepyz]
		vz = Uq[:, :, svz]

		ones = np.empty_like(vz)
		# print(ones.shape)
		ones[:,:] = 1.0

		''' Unpack '''
		mu = self.mu*ones
		rho = self.rho*ones

		''' Flag non-physical state '''
		if flag_non_physical:
			if np.any(mu < 0.):
				raise errors.NotPhysicalError
			if np.any(rho < 0.):
				raise errors.NotPhysicalError

		''' Nested functions for common quantities '''
		def get_waveSpeed():
			varq = np.sqrt(mu/rho)
			if flag_non_physical:
				if np.any(varq < 0.):
					raise errors.NotPhysicalError
			return varq

		''' Compute '''
		vname = self.AdditionalVariables[var_name].name

		if vname is self.AdditionalVariables["WaveSpeed"].name:
			varq = get_waveSpeed()
		elif vname is self.AdditionalVariables["MaxWaveSpeed"].name:
			varq = get_waveSpeed()
		else:
			raise NotImplementedError

		return varq

	def compute_pressure_gradient(self, Uq, grad_Uq):
		'''
		Compute the gradient of pressure with respect to physical space. This is
		needed for pressure-based shock sensors.

		Inputs:
		-------
			Uq: solution in each element evaluated at quadrature points
			[ne, nq, ns]
			grad_Uq: gradient of solution in each element evaluted at quadrature
				points [ne, nq, ns, ndims]

		Outputs:
		--------
			array: gradient of pressure with respected to physical space
				[ne, nq, ndims]
		'''
		srho = self.get_state_slice("Density")
		srhoE = self.get_state_slice("Energy")
		smom = self.get_momentum_slice()
		rho = Uq[:, :, srho]
		rhoE = Uq[:, :, srhoE]
		mom = Uq[:, :, smom]
		gamma = self.gamma

		# Compute dp/dU
		dpdU = np.empty_like(Uq)
		dpdU[:, :, srho] = (.5 * (gamma - 1) * np.sum(mom**2, axis = 2,
			keepdims=True) / rho)
		dpdU[:, :, smom] = (1 - gamma) * mom / rho
		dpdU[:, :, srhoE] = gamma - 1

		# Multiply with dU/dx
		return np.einsum('ijk, ijkl -> ijl', dpdU, grad_Uq)


class Antiplane(AntiplaneWave):
	'''
	This class corresponds to 2D Euler equations for a calorically
	perfect gas. It inherits attributes and methods from the Euler class.
	See Euler for detailed comments of attributes and methods.

	Additional methods and attributes are commented below.
	'''
	NUM_STATE_VARS = 3
	NDIMS = 2

	def __init__(self):
		super().__init__()

	def set_maps(self):
		super().set_maps()

		d = {
			antiWave_fcn_type.PlaneSine: antiWave_fcns.PlaneSine,
			antiWave_fcn_type.Zeros: antiWave_fcns.Zeros,
		}

		self.IC_fcn_map.update(d)
		self.exact_fcn_map.update(d)
		self.BC_fcn_map.update(d)

		self.source_map.update({
			antiWave_source_type.PointSource : antiWave_fcns.PointSource,
		})

		# self.conv_num_flux_map.update({
		# 	base_conv_num_flux_type.LaxFriedrichs :
		# 		antiWave_fcns.LaxFriedrichs2D,
		# 	#TODO: Add upwind flux
		# })

	class StateVariables(Enum):
		StrainZX = "\\varepsilon zx"
		StrainYZ = "\\varepsilon yx"
		VelocityZ = "v z"

	def get_state_indices(self):
		iepzx = self.get_state_index("StrainZX")
		iepyz = self.get_state_index("StrainYZ")
		ivz = self.get_state_index("VelocityZ")
		return iepzx, iepyz, ivz

	def get_state_slices(self):
		iepzx = self.get_state_slice("StrainZX")
		iepyz = self.get_state_slice("StrainYZ")
		ivz = self.get_state_slice("VelocityZ")

		return iepzx, iepyz, ivz

	# def get_momentum_slice(self):
	# 	irhou = self.get_state_index("XMomentum")
	# 	irhov = self.get_state_index("YMomentum")
	# 	smom = slice(irhou, irhov + 1)

	# 	return smom

	def get_sources_res(self, basis, physics, mesh, elem_helpers, time):
		'''
		This method directly compute point source related residuals

		Inputs:
		-------
			x: quadrature points coordinates
			A0: point source amplitude

		Outputs:
		--------
		    self: attributes initialized
		'''
		djac_elems = elem_helpers.djac_elems # [ne, nq, 1]

		sources = physics.point_sources

		nt = np.size(sources[0].data)
		dt = sources[0].dt
		times_data = np.linspace(0,(nt-1)*dt,nt)
		source_data = sources[0].data
		# interpolate source data in time_data to time
		source_amp = np.interp(time, times_data, source_data)
		x_source = sources[0].xs

		# print("source at: ", x_source)

		eId = sources[0].ele_ID
		# print("source in: ", mesh.elements[eId].node_coords)

		x_elems = elem_helpers.x_elems[eId,:,:] # [nq, nd]

		x_verts = np.array([[1,0], [0,1], [0,0]])
		# print("x_verts has shape: ", x_verts.shape)
		x_phys = mesh_tools.ref_to_phys(mesh, eId, x_verts)
		new_basis = cpObj.deepcopy(basis)
		new_basis.get_basis_val_grads(x_verts, get_val=True)
		verts_val = new_basis.basis_val # [3, nb]
		# print("verts_val has shape: ", verts_val.shape)

		basis_val = elem_helpers.basis_val # [nq, nb]

		x_knowns = np.append(x_phys, x_elems, axis=0)
		basis_knowns = np.append(verts_val, basis_val, axis=0)
		# print("x_knowns has shape: ", x_knowns.shape)
		# print("basis_knowns has shape: ", basis_knowns.shape)

		# print("interpolated in x_knowns: ", eId, " from ", x_knowns)
		# print("interpolated from basis_knowns: ",basis_knowns)

		# get the basis function values at the source location
		interpolator = LinearNDInterpolator(x_knowns, basis_knowns)
		basis_values_at_sou = interpolator(x_source) # [nb,]
		# print("basis_values_at_sou has shape: ", basis_values_at_sou.shape)

		res = np.zeros([elem_helpers.x_elems.shape[0], \
				basis_val.shape[1], physics.NUM_STATE_VARS])

		iepzx, iepyz, ivz = physics.get_state_indices()

		# print(basis_values_at_sou)

		# res[eId,:,iepzx] = source_amp * basis_values_at_sou
		res[eId,:,ivz] = source_amp * basis_values_at_sou

		return res # [ne, nb, ns]

	def get_conv_flux_interior(self, Uq):
		# Get indices/slices of state variables
		iepzx, iepyz, ivz = self.get_state_indices()

		# Get state variables
		epzx = Uq[:, :, iepzx]
		epyz = Uq[:, :, iepyz]
		vz = Uq[:, :, ivz]

		mu = self.mu
		rho = self.rho

		# Assemble flux matrix
		F = np.empty(Uq.shape + (self.NDIMS,)) # [n, nq, ns, ndims]
		F[:, :, iepzx, 0] = -0.5*vz
		F[:, :, iepzx, 1] = 0.
		F[:, :, iepyz, 0] = 0.
		F[:, :, iepyz, 1] = -0.5*vz
		F[:, :, ivz, 0] = -2.0*mu/rho*epzx
		F[:, :, ivz, 1] = -2.0*mu/rho*epyz

		return F, (0.5*vz, 0.5*vz, -2.0*mu/rho*epzx, -2.0*mu/rho*epyz)
