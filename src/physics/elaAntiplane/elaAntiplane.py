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

import errors
import general

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
	'''
	PHYSICS_TYPE = general.PhysicsType.ElaAntiplain

	def __init__(self):
		super().__init__()
		self.mu = 0.
		self.rho = 0.

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
			GasConstant: mass-specific gas constant
			SpecificHeatRatio: ratio of specific heats

		Outputs:
		--------
			self: physical parameters set
		'''
		self.mu = ShearModulus
		self.rho = Density

	class AdditionalVariables(Enum):
		# StateVariable = "\\Psi"
		WaveSpeed = "cs"
		MaxWaveSpeed = "\\lambda"
		# SlipRate = "|V|"
		# ZDisplacement = "D"

	def compute_additional_variable(self, var_name, Uq, flag_non_physical):
		''' Extract state variables '''
		sepzx = self.get_state_slice("StrainZX")
		sepyz = self.get_state_slice("StrainZX")
		svz = self.get_state_slice("StrainZX")
		epzx = Uq[:, :, sepzx]
		epyz = Uq[:, :, sepyz]
		vz = Uq[:, :, svz]

		''' Unpack '''
		mu = self.mu
		rho = self.rho

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
			varq = get_waveSpeed
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
		}

		self.IC_fcn_map.update(d)
		self.exact_fcn_map.update(d)
		self.BC_fcn_map.update(d)

		self.source_map.update({
			euler_source_type.StiffFriction : euler_fcns.StiffFriction,
			euler_source_type.TaylorGreenSource :
					euler_fcns.TaylorGreenSource,
			euler_source_type.GravitySource : euler_fcns.GravitySource,
		})

		self.conv_num_flux_map.update({
			base_conv_num_flux_type.LaxFriedrichs :
				euler_fcns.LaxFriedrichs2D,
			euler_conv_num_flux_type.Roe : euler_fcns.Roe2D,
		})

	class StateVariables(Enum):
		StrainZX = "\\varepsilon zx"
		StrainYZ = "\\varepsilon yx"
		VelocityZ = "v z"

	def get_state_indices(self):
		iepzx = self.get_state_index("StrainZX")
		iepyz = self.get_state_index("StrainYZ")
		ivz = self.get_state_index("VelocityZ")

		return iepzx, iepyz, ivz

	# def get_momentum_slice(self):
	# 	irhou = self.get_state_index("XMomentum")
	# 	irhov = self.get_state_index("YMomentum")
	# 	smom = slice(irhou, irhov + 1)

	# 	return smom

	def get_conv_flux_interior(self, Uq):
		# Get indices/slices of state variables
		iepzx, iepyz, ivz = self.get_state_indices()

		# Get state variables
		epzx = Uq[:, :, iepzx]
		epyz = Uq[:, :, iepyz]
		vz = Uq[:, :, ivz]

		# Assemble flux matrix
		F = np.empty(Uq.shape + (self.NDIMS,)) # [n, nq, ns, ndims]
		F[:,:,irho,  :] = mom          # Flux of mass in all directions
		F[:,:,irhou, 0] = rho * u2 + p # x-flux of x-momentum
		F[:,:,irhov, 0] = rhouv        # x-flux of y-momentum
		F[:,:,irhou, 1] = rhouv        # y-flux of x-momentum
		F[:,:,irhov, 1] = rho * v2 + p # y-flux of y-momentum
		F[:,:,irhoE, 0] = H * u        # x-flux of energy
		F[:,:,irhoE, 1] = H * v        # y-flux of energy

		return F, (u2, v2, rho, p)
