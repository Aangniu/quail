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
	SlipWall = auto()
	PressureOutlet = auto()


class SourceType(Enum):
	'''
	Enum class that stores the types of source terms. These
	source terms are specific to the available Euler equation sets.
	'''
	StiffFriction = auto()
	TaylorGreenSource = auto()
	GravitySource = auto()


class ConvNumFluxType(Enum):
	'''
	Enum class that stores the types of convective numerical fluxes. These
	numerical fluxes are specific to the available Euler equation sets.
	'''
	Roe = auto()


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
		sepzx, sepyz, svz = physics.get_state_slices()

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

		vz = np.sin(2*np.pi * k_wlength * x[:,:,0])

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

class SlipWall(BCWeakPrescribed):
	'''
	This class corresponds to a slip wall. See documentation for more
	details.
	'''
	def get_boundary_state(self, physics, UqI, normals, x, t):
		smom = physics.get_momentum_slice()

		# Unit normals
		n_hat = normals/np.linalg.norm(normals, axis=2, keepdims=True)

		# Remove momentum contribution in normal direction from boundary
		# state
		rhoveln = np.sum(UqI[:, :, smom] * n_hat, axis=2, keepdims=True)
		UqB = UqI.copy()
		UqB[:, :, smom] -= rhoveln * n_hat

		return UqB


class PressureOutlet(BCWeakPrescribed):
	'''
	This class corresponds to an outflow boundary condition with static
	pressure prescribed. See documentation for more details.

	Attributes:
	-----------
	p: float
		pressure
	'''
	def __init__(self, p):
		'''
		This method initializes the attributes.

		Inputs:
		-------
			p: pressure

		Outputs:
		--------
		    self: attributes initialized
		'''
		self.p = p

	def get_boundary_state(self, physics, UqI, normals, x, t):
		# Unpack
		srho = physics.get_state_slice("Density")
		srhoE = physics.get_state_slice("Energy")
		smom = physics.get_momentum_slice()

		# Pressure
		pB = self.p

		gamma = physics.gamma

		UqB = UqI.copy()

		# Unit normals
		n_hat = normals/np.linalg.norm(normals, axis=2, keepdims=True)

		# Interior velocity in normal direction
		rhoI = UqI[:, :, srho]
		velI = UqI[:, :, smom]/rhoI
		velnI = np.sum(velI*n_hat, axis=2, keepdims=True)

		if np.any(velnI < 0.):
			print("Incoming flow at outlet")

		# Interior pressure
		pI = physics.compute_variable("Pressure", UqI)

		if np.any(pI < 0.):
			raise errors.NotPhysicalError

		# Interior speed of sound
		cI = physics.compute_variable("SoundSpeed", UqI)
		JI = velnI + 2.*cI/(gamma - 1.)
		# Interior velocity in tangential direction
		veltI = velI - velnI*n_hat

		# Normal Mach number
		Mn = velnI/cI
		if np.any(Mn >= 1.):
			# If supersonic, then extrapolate interior to exterior
			return UqB

		# Boundary density from interior entropy
		rhoB = rhoI*np.power(pB/pI, 1./gamma)
		UqB[:, :, srho] = rhoB

		# Boundary speed of sound
		cB = np.sqrt(gamma*pB/rhoB)
		# Boundary velocity
		velB = (JI - 2.*cB/(gamma-1.))*n_hat + veltI
		UqB[:, :, smom] = rhoB*velB

		# Boundary energy
		rhovel2B = rhoB*np.sum(velB**2., axis=2, keepdims=True)
		UqB[:, :, srhoE] = pB/(gamma - 1.) + 0.5*rhovel2B

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

class StiffFriction(SourceBase):
	'''
	Stiff source term (1D) of the form:
	S = [0, nu*rho*u, nu*rho*u^2]

	Attributes:
	-----------
	nu: float
		stiffness parameter
	'''
	def __init__(self, nu=-1, **kwargs):
		super().__init__(kwargs)
		'''
		This method initializes the attributes.

		Inputs:
		-------
			nu: source term parameter

		Outputs:
		--------
		    self: attributes initialized
		'''
		self.nu = nu

	def get_source(self, physics, Uq, x, t):
		nu = self.nu

		irho, irhou, irhoE = physics.get_state_indices()

		S = np.zeros_like(Uq)

		eps = general.eps
		S[:, :, irho] = 0.0
		S[:, :, irhou] = nu*(Uq[:, :, irhou])
		S[:, :, irhoE] = nu*((Uq[:, :, irhou])**2/(eps + Uq[:, :, irho]))

		return S

	def get_jacobian(self, physics, Uq, x, t):
		nu = self.nu

		irho, irhou, irhoE = physics.get_state_indices()

		jac = np.zeros([Uq.shape[0], Uq.shape[1], Uq.shape[-1], Uq.shape[-1]])
		vel = Uq[:, :, 1]/(general.eps + Uq[:, :, 0])

		jac[:, :, irhou, irhou] = nu
		jac[:, :, irhoE, irho] = -nu*vel**2
		jac[:, :, irhoE, irhou] = 2.0*nu*vel

		return jac


class TaylorGreenSource(SourceBase):
	'''
	Source term for 2D Taylor-Green vortex (see above). Reference:
		[1] C. Wang, "Reconstructed discontinous Galerkin method for the
		compressible Navier-Stokes equations in arbitrary Langrangian and
		Eulerian formulation", PhD Thesis, North Carolina State University,
		2017.
	'''
	def get_source(self, physics, Uq, x, t):
		gamma = physics.gamma

		irho, irhou, irhov, irhoE = physics.get_state_indices()

		S = np.zeros_like(Uq)

		S[:, :, irhoE] = np.pi/(4.*(gamma - 1.))*(np.cos(3.*np.pi*x[:, :, 0])*
				np.cos(np.pi*x[:, :, 1]) - np.cos(np.pi*x[:, :, 0])*np.cos(3.*
				np.pi*x[:, :, 1]))

		return S


class GravitySource(SourceBase):
	'''
	Gravity source term used with the GravityRiemann problem defined above.
	Adds gravity to the inviscid Euler equations. See the following reference
	for further details:
		[1] X. Zhang, C.-W. Shu, "Positivity-preserving high-order
		discontinuous Galerkin schemes for compressible Euler equations
		with source terms, Journal of Computational Physics 230
		(2011) 1238–1248.
	'''
	def __init__(self, gravity=0., **kwargs):
		super().__init__(kwargs)
		'''
		This method initializes the attributes.

		Inputs:
		-------
			gravity: gravity constant

		Outputs:
		--------
		    self: attributes initialized
		'''
		self.gravity = gravity

	def get_source(self, physics, Uq, x, t):
		# Unpack
		gamma = physics.gamma
		g = self.gravity

		irho, irhou, irhov, irhoE = physics.get_state_indices()

		S = np.zeros_like(Uq)

		rho = Uq[:, :, irho]
		rhov = Uq[:, :, irhov]

		S[:, :, irhov] = -rho * g
		S[:, :, irhoE] = -rhov * g

		return S # [ne, nq, ns]


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
class LaxFriedrichs1D(ConvNumFluxBase):
	'''
	This class corresponds to the local Lax-Friedrichs flux function for the
	Euler1D class. This replaces the generalized, less efficient version of
	the Lax-Friedrichs flux found in base.
	'''
	def compute_flux(self, physics, UqL, UqR, normals):
		# Normalize the normal vectors
		n_mag = np.linalg.norm(normals, axis=2, keepdims=True)
		n_hat = normals/n_mag

		# Left flux
		FqL, (u2L, rhoL, pL) = physics.get_conv_flux_projected(UqL, n_hat)

		# Right flux
		FqR, (u2R, rhoR, pR) = physics.get_conv_flux_projected(UqR, n_hat)

		# Jump
		dUq = UqR - UqL

		# Max wave speeds at each point
		aL = np.empty(pL.shape + (1,))
		aR = np.empty(pR.shape + (1,))
		aL[:, :, 0] = np.sqrt(u2L) + np.sqrt(physics.gamma * pL / rhoL)
		aR[:, :, 0] = np.sqrt(u2R) + np.sqrt(physics.gamma * pR / rhoR)
		idx = aR > aL
		aL[idx] = aR[idx]

		# Put together
		return 0.5 * n_mag * (FqL + FqR - aL*dUq)


class LaxFriedrichs2D(ConvNumFluxBase):
	'''
	This class corresponds to the local Lax-Friedrichs flux function for the
	Euler2D class. This replaces the generalized, less efficient version of
	the Lax-Friedrichs flux found in base.
	'''
	def compute_flux(self, physics, UqL, UqR, normals):
		# Normalize the normal vectors
		n_mag = np.linalg.norm(normals, axis=2, keepdims=True)
		n_hat = normals/n_mag

		# Left flux
		FqL, (u2L, v2L, rhoL, pL) = physics.get_conv_flux_projected(UqL,
				n_hat)

		# Right flux
		FqR, (u2R, v2R, rhoR, pR) = physics.get_conv_flux_projected(UqR,
				n_hat)

		# Jump
		dUq = UqR - UqL

		# Max wave speeds at each point
		aL = np.empty(pL.shape + (1,))
		aR = np.empty(pR.shape + (1,))
		aL[:, :, 0] = np.sqrt(u2L + v2L) + np.sqrt(physics.gamma * pL / rhoL)
		aR[:, :, 0] = np.sqrt(u2R + v2R) + np.sqrt(physics.gamma * pR / rhoR)
		idx = aR > aL
		aL[idx] = aR[idx]

		# Put together
		return 0.5 * n_mag * (FqL + FqR - aL*dUq)


class Roe1D(ConvNumFluxBase):
	'''
	1D Roe numerical flux. References:
		[1] P. L. Roe, "Approximate Riemann solvers, parameter vectors, and
		difference schemes," Journal of Computational Physics,
		43(2):357–372, 1981.
		[2] J. S. Hesthaven, T. Warburton, "Nodal discontinuous Galerkin
		methods: algorithms, analysis, and applications," Springer Science
		& Business Media, 2007.

	Attributes:
	-----------
	UqL: numpy array
		helper array for left state [nf, nq, ns]
	UqR: numpy array
		helper array for right state [nf, nq, ns]
	vel: numpy array
		helper array for velocity [nf, nq, ndims]
	alphas: numpy array
		helper array: left eigenvectors multipled by dU [nf, nq, ns]
	evals: numpy array
		helper array for eigenvalues [nf, nq, ns]
	R: numpy array
		helper array for right eigenvectors [nf, nq, ns, ns]
	'''
	def __init__(self, Uq=None):
		'''
		This method initializes the attributes.

		Inputs:
		-------
			Uq: values of the state variables (typically at the quadrature
				points) [nf, nq, ns]; used to allocate helper arrays; if None,
				then empty arrays allocated

		Outputs:
		--------
		    self: attributes initialized
		'''
		if Uq is not None:
			n = Uq.shape[0]
			nq = Uq.shape[1]
			ns = Uq.shape[-1]
			ndims = ns - 2
		else:
			n = nq = ns = ndims = 0

		self.UqL = np.zeros_like(Uq)
		self.UqR = np.zeros_like(Uq)
		self.vel = np.zeros([n, nq, ndims])
		self.alphas = np.zeros_like(Uq)
		self.evals = np.zeros_like(Uq)
		self.R = np.zeros([n, nq, ns, ns])

	def rotate_coord_sys(self, smom, Uq, n):
		'''
		This method expresses the momentum vector in the rotated coordinate
		system, which is aligned with the face normal and tangent.

		Inputs:
		-------
			smom: momentum slice
			Uq: values of the state variable (typically at the quadrature
				points) [nf, nq, ns]
			n: normals (typically at the quadrature points) [nf, nq, ndims]

		Outputs:
		--------
		    Uq: momentum terms modified
		'''
		Uq[:, :, smom] *= n

		return Uq

	def undo_rotate_coord_sys(self, smom, Uq, n):
		'''
		This method expresses the momentum vector in the standard coordinate
		system. It "undoes" the rotation above.

		Inputs:
		-------
			smom: momentum slice
			Uq: values of the state variable (typically at the quadrature
				points) [nf, nq, ns]
			n: normals (typically at the quadrature points) [nf, nq, ndims]

		Outputs:
		--------
		    Uq: momentum terms modified
		'''
		Uq[:, :, smom] /= n

		return Uq

	def roe_average_state(self, physics, srho, velL, velR, UqL, UqR):
		'''
		This method computes the Roe-averaged variables.

		Inputs:
		-------
			physics: physics object
			srho: density slice
			velL: left velocity (typically evaluated at the quadrature
				points) [nf, nq, ndims]
			velR: right velocity (typically evaluated at the quadrature
				points) [nf, nq, ndims]
			UqL: left state (typically evaluated at the quadrature
				points) [nf, nq, ns]
			UqR: right state (typically evaluated at the quadrature
				points) [nf, nq, ns]

		Outputs:
		--------
		    rhoRoe: Roe-averaged density [nf, nq, 1]
		    velRoe: Roe-averaged velocity [nf, nq, ndims]
		    HRoe: Roe-averaged total enthalpy [nf, nq, 1]
		'''
		rhoL_sqrt = np.sqrt(UqL[:, :, srho])
		rhoR_sqrt = np.sqrt(UqR[:, :, srho])
		HL = physics.compute_variable("TotalEnthalpy", UqL)
		HR = physics.compute_variable("TotalEnthalpy", UqR)

		velRoe = (rhoL_sqrt*velL + rhoR_sqrt*velR)/(rhoL_sqrt+rhoR_sqrt)
		HRoe = (rhoL_sqrt*HL + rhoR_sqrt*HR)/(rhoL_sqrt+rhoR_sqrt)
		rhoRoe = rhoL_sqrt*rhoR_sqrt

		return rhoRoe, velRoe, HRoe

	def get_differences(self, physics, srho, velL, velR, UqL, UqR):
		'''
		This method computes velocity, density, and pressure jumps.

		Inputs:
		-------
			physics: physics object
			srho: density slice
			velL: left velocity (typically evaluated at the quadrature
				points) [nf, nq, ndims]
			velR: right velocity (typically evaluated at the quadrature
				points) [nf, nq, ndims]
			UqL: left state (typically evaluated at the quadrature
				points) [nf, nq, ns]
			UqR: right state (typically evaluated at the quadrature
				points) [nf, nq, ns]

		Outputs:
		--------
		    drho: density jump [nf, nq, 1]
		    dvel: velocity jump [nf, nq, ndims]
		    dp: pressure jump [nf, nq, 1]
		'''
		dvel = velR - velL
		drho = UqR[:, :, srho] - UqL[:, :, srho]
		dp = physics.compute_variable("Pressure", UqR) - \
				physics.compute_variable("Pressure", UqL)

		return drho, dvel, dp

	def get_alphas(self, c, c2, dp, dvel, drho, rhoRoe):
		'''
		This method computes alpha_i = ith left eigenvector * dU.

		Inputs:
		-------
			c: speed of sound [nf, nq, 1]
			c2: speed of sound squared [nf, nq, 1]
			dp: pressure jump [nf, nq, 1]
			dvel: velocity jump [nf, nq, ndims]
			drho: density jump [nf, nq, 1]
			rhoRoe: Roe-averaged density [nf, nq, 1]

		Outputs:
		--------
		    alphas: left eigenvectors multipled by dU [nf, nq, ns]
		'''
		alphas = self.alphas

		alphas[:, :, 0:1] = 0.5/c2*(dp - c*rhoRoe*dvel[:, :, 0:1])
		alphas[:, :, 1:2] = drho - dp/c2
		alphas[:, :, -1:] = 0.5/c2*(dp + c*rhoRoe*dvel[:, :, 0:1])

		return alphas

	def get_eigenvalues(self, velRoe, c):
		'''
		This method computes the eigenvalues.

		Inputs:
		-------
			velRoe: Roe-averaged velocity [nf, nq, ndims]
			c: speed of sound [nf, nq, 1]

		Outputs:
		--------
		    evals: eigenvalues [nf, nq, ns]
		'''
		evals = self.evals

		evals[:, :, 0:1] = velRoe[:, :, 0:1] - c
		evals[:, :, 1:2] = velRoe[:, :, 0:1]
		evals[:, :, -1:] = velRoe[:, :, 0:1] + c

		return evals

	def get_right_eigenvectors(self, c, evals, velRoe, HRoe):
		'''
		This method computes the right eigenvectors.

		Inputs:
		-------
			c: speed of sound [nf, nq, 1]
			evals: eigenvalues [nf, nq, ns]
			velRoe: Roe-averaged velocity [nf, nq, ndims]
			HRoe: Roe-averaged total enthalpy [nf, nq, 1]

		Outputs:
		--------
		    R: right eigenvectors [nf, nq, ns, ns]
		'''
		R = self.R

		# first row
		R[:, :, 0, 0:2] = 1.; R[:, :, 0, -1] = 1.
		# second row
		R[:, :, 1, 0] = evals[:, :, 0]; R[:, :, 1, 1] = velRoe[:, :, 0]
		R[:, :, 1, -1] = evals[:, :, -1]
		# last row
		R[:, :, -1, 0:1] = HRoe - velRoe[:, :, 0:1]*c;
		R[:, :, -1, 1:2] = 0.5*np.sum(velRoe*velRoe, axis=2, keepdims=True)
		R[:, :, -1, -1:] = HRoe + velRoe[:, :, 0:1]*c

		return R

	def compute_flux(self, physics, UqL_std, UqR_std, normals):
		# Reshape arrays
		n = UqL_std.shape[0]
		nq = UqL_std.shape[1]
		ns = UqL_std.shape[2]
		ndims = ns - 2
		self.UqL_stdL = np.zeros_like(UqL_std)
		self.UqL_stdR = np.zeros_like(UqL_std)
		self.vel = np.zeros([n, nq, ndims])
		self.alphas = np.zeros_like(UqL_std)
		self.evals = np.zeros_like(UqL_std)
		self.R = np.zeros([n, nq, ns, ns])

		# Unpack
		srho = physics.get_state_slice("Density")
		smom = physics.get_momentum_slice()
		gamma = physics.gamma

		# Unit normals
		n_mag = np.linalg.norm(normals, axis=2, keepdims=True)
		n_hat = normals/n_mag

		# Copy values from standard coordinate system before rotating
		UqL = UqL_std.copy()
		UqR = UqR_std.copy()

		# Rotated coordinate system
		UqL = self.rotate_coord_sys(smom, UqL, n_hat)
		UqR = self.rotate_coord_sys(smom, UqR, n_hat)

		# Velocities
		velL = UqL[:, :, smom]/UqL[:, :, srho]
		velR = UqR[:, :, smom]/UqR[:, :, srho]

		# Roe-averaged state
		rhoRoe, velRoe, HRoe = self.roe_average_state(physics, srho, velL,
				velR, UqL, UqR)

		# Speed of sound from Roe-averaged state
		c2 = (gamma - 1.)*(HRoe - 0.5*np.sum(velRoe*velRoe, axis=2,
				keepdims=True))
		if np.any(c2 <= 0.):
			# Non-physical state
			raise errors.NotPhysicalError
		c = np.sqrt(c2)

		# Jumps
		drho, dvel, dp = self.get_differences(physics, srho, velL, velR,
				UqL, UqR)

		# alphas (left eigenvectors multiplied by dU)
		alphas = self.get_alphas(c, c2, dp, dvel, drho, rhoRoe)

		# Eigenvalues
		evals = self.get_eigenvalues(velRoe, c)

		# Entropy fix (currently commented as we have yet to decide
		# if this is needed long term)
		# eps = np.zeros_like(evals)
		# eps[:, :, :] = (1e-2 * c)
		# fix = np.argwhere(np.logical_and(evals < eps, evals > -eps))
		# fix_shape = fix[:, 0], fix[:, 1], fix[:, 2]
		# evals[fix_shape] = 0.5 * (eps[fix_shape] + evals[fix_shape]* \
		# 	evals[fix_shape] / eps[fix_shape])

		# Right eigenvector matrix
		R = self.get_right_eigenvectors(c, evals, velRoe, HRoe)

		# Form flux Jacobian matrix multiplied by dU
		FRoe = np.einsum('ijkl, ijl -> ijk', R, np.abs(evals)*alphas)

		# Undo rotation
		FRoe = self.undo_rotate_coord_sys(smom, FRoe, n_hat)

		# Left flux
		FL, _ = physics.get_conv_flux_projected(UqL_std, n_hat)

		# Right flux
		FR, _ = physics.get_conv_flux_projected(UqR_std, n_hat)

		return .5*n_mag*(FL + FR - FRoe) # [nf, nq, ns]


class Roe2D(Roe1D):
	'''
	2D Roe numerical flux. This class inherits from the Roe1D class.
	See Roe1D for detailed comments on the attributes and methods.
	In this class, several methods are updated to account for the extra
	dimension.
	'''
	def rotate_coord_sys(self, smom, Uq, n):
		vel = self.vel
		vel[:] = Uq[:, :, smom]

		vel[:, :, 0] = np.sum(Uq[:, :, smom]*n, axis=2)
		vel[:, :, 1] = np.sum(Uq[:, :, smom]*n[:, :, ::-1]*np.array([[-1.,
				1.]]), axis=2)

		Uq[:, :, smom] = vel

		return Uq

	def undo_rotate_coord_sys(self, smom, Uq, n):
		vel = self.vel
		vel[:] = Uq[:, :, smom]

		vel[:, :, 0] = np.sum(Uq[:, :, smom]*n*np.array([[1., -1.]]), axis=2)
		vel[:, :, 1] = np.sum(Uq[:, :, smom]*n[:, :, ::-1], axis=2)

		Uq[:, :, smom] = vel

		return Uq

	def get_alphas(self, c, c2, dp, dvel, drho, rhoRoe):
		alphas = self.alphas

		alphas = super().get_alphas(c, c2, dp, dvel, drho, rhoRoe)

		alphas[:, :, 2:3] = rhoRoe*dvel[:, :, -1:]

		return alphas

	def get_eigenvalues(self, velRoe, c):
		evals = self.evals

		evals = super().get_eigenvalues(velRoe, c)

		evals[:, :, 2:3] = velRoe[:, :, 0:1]

		return evals

	def get_right_eigenvectors(self, c, evals, velRoe, HRoe):
		R = self.R

		R = super().get_right_eigenvectors(c, evals, velRoe, HRoe)

		i = 2

		# First row
		R[:, :, 0, i] = 0.
		#  Second row
		R[:, :, 1, i] = 0.
		#  Last (fourth) row
		R[:, :, -1, i] = velRoe[:, :, -1]
		#  Third row
		R[:, :, i, 0] = velRoe[:, :, -1];  R[:, :, i, 1] = velRoe[:, :, -1]
		R[:, :, i, -1] = velRoe[:, :, -1]; R[:, :, i, i] = 1.

		return R