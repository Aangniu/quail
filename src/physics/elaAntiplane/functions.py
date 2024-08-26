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
#       File : src/physics/elaAntiplane/functions.py
#
#       Contains definitions of Functions, boundary conditions, and source
#       terms for the anti-plane 2D elastic wave equations.
#
# ------------------------------------------------------------------------ #
from enum import Enum, auto
import numpy as np
from scipy.optimize import fsolve, root

import errors
import general

from physics.base.data import (FcnBase, BCWeakRiemann, BCWeakPrescribed,
        SourceBase, ConvNumFluxBase)


class FcnType(Enum):
	'''
	Enum class that stores the types of analytical functions for initial
	conditions, exact solutions, and/or boundary conditions. These
	functions are specific to the available anti-plane elastic wave equation
	sets.
	'''
	PlaneSine = auto()

class BCType(Enum):
	'''
	Enum class that stores the types of boundary conditions. These
	boundary conditions are specific to the available Euler equation sets.
	'''
	Absorbing = auto()


class SourceType(Enum):
	'''
	Enum class that stores the types of source terms. These
	source terms are specific to the available Euler equation sets.
	'''
	PointSource = auto()


# class ConvNumFluxType(Enum):
# 	'''
# 	Enum class that stores the types of convective numerical fluxes. These
# 	numerical fluxes are specific to the available Euler equation sets.
# 	'''
# 	Roe = auto()


'''
---------------
State functions
---------------
These classes inherit from the FcnBase class. See FcnBase for detailed
comments of attributes and methods. Information specific to the
corresponding child classes can be found below. These classes should
correspond to the FcnType enum members above.
'''

class PlaneSine(FcnBase):
	'''
	An initial value problem for plane waves

	Attributes:
	-----------
	epzx: float
		base strain-zx
	epyz: float
		base strain-yz
	vz: float
		base z-velocity
	'''
	def __init__(self, epzx=1., epyz=1., vz=1.):
		'''
		This method initializes the attributes.

		Inputs:
		-------
			epzx: base strain-zx
			epyz: base strain-yz
			vz: base z-velocity

		Outputs:
		--------
		    self: attributes initialized
		'''
		self.epzx = epzx
		self.epyz = epyz
		self.vz = vz

	def get_state(self, physics, x, t):
		Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])
		mu = physics.mu
		rho = physics.rho
		sepzx, sepyz, svz = physics.get_state_indices()

		''' Base flow '''
		# strain-zx
		epzx = self.epzx
		# strain-yz
		epyz = self.epyz
		# z-velocity
		vz = self.vz

		k_wlength = 1.0

		epzx = np.zeros_like(x[:,:,0])
		epyz = np.zeros_like(x[:,:,0])
		vz = np.zeros_like(x[:,:,0])

		condition = np.abs(x[:, :, 0]) < 0.5
		vz[condition] = np.sin(2*np.pi * k_wlength * x[:,:,0][condition])
		epzx[condition] = -0.5 / 1.0 * vz[condition]

		Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])
		Uq[:, :, sepzx] = epzx
		Uq[:, :, sepyz] = epyz
		Uq[:, :, svz] = vz

		return Uq # [ne, nq, ns]

'''
-------------------
Boundary conditions
-------------------
These classes inherit from either the BCWeakRiemann or BCWeakPrescribed
classes. See those parent classes for detailed comments of attributes
and methods. Information specific to the corresponding child classes can be
found below. These classes should correspond to the BCType enum members
above.
'''

class Absorbing(BCWeakPrescribed):
	'''
	This class corresponds to a perfectly matching layer.
	See documentation for more details.
	'''
	def get_boundary_state(self, physics, UqI, normals, x, t):
		UqB = UqI.copy()
		UqB[:, :, :] = 0.0
		return UqB


'''
---------------------
Source term functions
---------------------
These classes inherit from the SourceBase class. See SourceBase for detailed
comments of attributes and methods. Information specific to the
corresponding child classes can be found below. These classes should
correspond to the SourceType enum members above.
'''

class PointSource(SourceBase):
	'''
	Stiff source term (1D) of the form:
	S = [0, nu*rho*u, nu*rho*u^2]

	Attributes:
	-----------
	nu: float
		stiffness parameter
	'''
	def __init__(self, x0=np.array([0.5,0.5]), A0=0.0, **kwargs):
		super().__init__(kwargs)
		'''
		This method initializes the attributes.

		Inputs:
		-------
			x0: point source location
			A0: point source amplitude

		Outputs:
		--------
		    self: attributes initialized
		'''
		self.x0 = x0
		self.A0 = A0

	def get_source(self, physics, Uq, x, t):
		x0 = self.x0
		A0 = self.A0

		iepzx, iepyz, ivz = physics.get_state_indices()

		S = np.zeros_like(Uq)

		# Define gaussian-function point source
		# TODO: Use basis functions to define point source
		# Amplitude with a rise time centered at t0
		t0 = 0.5
		A = A0*np.exp(-np.sum((t - t0)**2)/0.01)
		S[:, :, ivz] = A*np.exp(-np.sum((x - x0)**2, axis=2)/0.01)
		S[:, :, iepzx] = 0.0
		S[:, :, iepyz] = 0.0

		return S

'''
------------------------
Numerical flux functions
------------------------
These classes inherit from the ConvNumFluxBase or DiffNumFluxBase class.
See ConvNumFluxBase/DiffNumFluxBase for detailed comments of attributes
and methods. Information specific to the corresponding child classes can
be found below. These classes should correspond to the ConvNumFluxType
or DiffNumFluxType enum members above.
'''

# class LaxFriedrichs2D(ConvNumFluxBase):
# 	'''
# 	This class corresponds to the local Lax-Friedrichs flux function for the
# 	Antiplane class. This replaces the generalized, less efficient version of
# 	the Lax-Friedrichs flux found in base.
# 	'''
# 	def compute_flux(self, physics, UqL, UqR, normals):
# 		# Normalize the normal vectors
# 		n_mag = np.linalg.norm(normals, axis=2, keepdims=True)
# 		n_hat = normals/n_mag

# 		# Left flux
# 		FqL, (u2L, v2L, rhoL, pL) = physics.get_conv_flux_projected(UqL,
# 				n_hat)

# 		# Right flux
# 		FqR, (u2R, v2R, rhoR, pR) = physics.get_conv_flux_projected(UqR,
# 				n_hat)

# 		# Jump
# 		dUq = UqR - UqL

# 		# Max wave speeds at each point
# 		aL = np.empty(pL.shape + (1,))
# 		aR = np.empty(pR.shape + (1,))
# 		aL[:, :, 0] = np.sqrt(u2L + v2L) + np.sqrt(physics.gamma * pL / rhoL)
# 		aR[:, :, 0] = np.sqrt(u2R + v2R) + np.sqrt(physics.gamma * pR / rhoR)
# 		idx = aR > aL
# 		aL[idx] = aR[idx]

# 		# Put together
# 		return 0.5 * n_mag * (FqL + FqR - aL*dUq)