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
	Zeros = auto()

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


class ConvNumFluxType(Enum):
	'''
	Enum class that stores the types of convective numerical fluxes. These
	numerical fluxes are specific to the elastic wave equation sets.
	'''
	DynamicRupture = auto()


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
		# vz[condition] = np.sin(2*np.pi * k_wlength * x[:,:,0][condition])
		vz[condition] = 1.0
		epzx[condition] = -0.5 / 1.0 * vz[condition]

		Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])
		Uq[:, :, sepzx] = epzx
		Uq[:, :, sepyz] = epyz
		Uq[:, :, svz] = vz

		return Uq # [ne, nq, ns]

class Zeros(FcnBase):
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

		epzx = np.zeros_like(x[:,:,0])
		epyz = np.zeros_like(x[:,:,0])
		vz = np.zeros_like(x[:,:,0])

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
	Point source term for the anti-plane elastic wave equation.

	Attributes:
	-----------
	x0: numpy array
		point source location
	A0: float
		point source amplitude
	'''
	def __init__(self, x0=np.array([0.0,0.0]), A0=0.0, **kwargs):
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
		t0 = 0.2
		A = A0*np.exp(-np.sum((t - t0)**2)/0.05)
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

class DynamicRupture(ConvNumFluxBase):
	def get_rotated_var(self, UqL, UqR, n_hat, s_hat, ns, inv=False):
		'''
		This method computes the rotation matrix to rotate the state vectors
		to the fault-aligned coordinate system.

		Inputs:
		-------
			n_hat: directions from left to right [nf, nq, ndims]
			s_hat: directions from left to right [nf, nq, ndims]

		Outputs:
		--------
			mat_T: rotation matrix [nf, ns, ns]
		'''
		if inv:
			nf = n_hat.shape[0]
			nq = n_hat.shape[1]
			mat_T = np.zeros([ns, ns])
			VqL = np.zeros([nf, nq, ns])
			VqR = np.zeros([nf, nq, ns])
			for i_f in range(nf):
				for i_q in range(nq):
					mat_T[0,0] = n_hat[i_f,i_q,0]
					mat_T[0,1] = s_hat[i_f,i_q,0]
					mat_T[1,0] = n_hat[i_f,i_q,1]
					mat_T[1,1] = s_hat[i_f,i_q,1]
					mat_T[2,2] = 1.0

					VqL[i_f,i_q,:] = mat_T.dot(UqL[i_f,i_q,:])
					VqR[i_f,i_q,:] = mat_T.dot(UqR[i_f,i_q,:])
					# print("n",n_hat[i_f,i_q,:], "l", s_hat[i_f,i_q,:])
					# print("V",VqL[i_f,i_q,:],"U",UqL[i_f,i_q,:])
		else:
			nf = n_hat.shape[0]
			nq = n_hat.shape[1]
			mat_T = np.zeros([ns, ns])
			VqL = np.zeros([nf, nq, ns])
			VqR = np.zeros([nf, nq, ns])
			for i_f in range(nf):
				for i_q in range(nq):
					mat_T[0,0] = n_hat[i_f,i_q,0]
					mat_T[1,0] = s_hat[i_f,i_q,0]
					mat_T[0,1] = n_hat[i_f,i_q,1]
					mat_T[1,1] = s_hat[i_f,i_q,1]
					mat_T[2,2] = 1.0

					VqL[i_f,i_q,:] = mat_T.dot(UqL[i_f,i_q,:])
					VqR[i_f,i_q,:] = mat_T.dot(UqR[i_f,i_q,:])
					# print("n",n_hat[i_f,i_q,:], "l", s_hat[i_f,i_q,:])
					# print("V",VqL[i_f,i_q,:],"U",UqL[i_f,i_q,:])

		return VqL, VqR


	'''
	This class corresponds to the DynamicRupture flux function for the
	Antiplane class. This is used to define frictional boundary conditions
	on the fault plane.

	Inputs:
	-------
	    normals: directions from left to right [nf, nq, ndims]
	'''
	def compute_flux(self, physics, UqL, UqR, normals,\
			parallels, stVars, slRates, oSV, oSR, dt, t, xs):
		# Step1: Rotate the state vectors to the fault-aligned coordinate
		# systems.
		## Normalize the normal vectors
		n_mag = np.linalg.norm(normals, axis=2, keepdims=True)
		n_hat = normals/n_mag
		## Normalize the fault parallel vectors
		s_mag = np.linalg.norm(parallels, axis=2, keepdims=True)
		s_hat = parallels/s_mag
		## Rotate state variables to face-aligned coords.
		ns = physics.NUM_STATE_VARS

		iepzx, iepyz, ivz = physics.get_state_indices()

		# Uq0 that stores initial "STRAIN" field, tau0/2*mu
		Uq0 = UqL.copy()
		epzyInit = 15.0

		Uq0[:, :, iepzx] = 0.0
		Uq0[:, :, iepyz] = epzyInit
		# nf = oSR.shape[0]
		# nq = oSR.shape[1]
		# for i_f in range(nf):
		# 	for i_q in range(nq):
		# 		if np.abs(xs[i_f,i_q,0]) < 0.5:
		# 			Uq0[i_f, i_q, iepyz] = epzyInit
		# 		else:
		# 			Uq0[i_f, i_q, iepyz] = epzyInit
		Uq0[:, :, ivz] = 0.0

		# UqNuc that stores nucleation "STRAIN" field, tauNuc/2*mu,
		# Note: this solution is added as part of the solution
		UqNuc = UqL.copy()

		UqNuc[:, :, iepzx] = 0.0
		nf = oSR.shape[0]
		nq = oSR.shape[1]
		for i_f in range(nf):
			for i_q in range(nq):
				if np.abs(xs[i_f,i_q,0]) < 1.0:
					UqNuc[i_f, i_q, iepyz] = np.minimum(t,0.1)/0.1*(3.0 - 3.0*np.abs(xs[i_f,i_q,0]))
				else:
					UqNuc[i_f, i_q, iepyz] = 0.0
		UqNuc[:, :, ivz] = 0.0

		Vq0, VqNuc = self.get_rotated_var(Uq0, UqNuc, n_hat, s_hat, ns)

		# print("before: ", UqL[0])
		VqL, VqR = self.get_rotated_var(UqL, UqR, n_hat, s_hat, ns)
		# print("rotated: ", VqL[0])
		# VqL, VqR = self.get_rotated_var(VqL, VqR, n_hat, s_hat, ns,True)
		# print("rotated back: ", VqL[0])

		# Step2: Compute the quantities needed for relationship between
		# slip rate V0 and shear stregth.
		# TODO: Add the material properties L and R to the physics object.
		muL = physics.mu; muR = physics.mu
		rhoL = physics.rho; rhoR = physics.rho

		csL = np.sqrt(muL/rhoL); csR = np.sqrt(muR/rhoR)

		etaS = 1.0/(csL/muL + csR/muR)

		# Get state variables
		epzxL = VqL[:, :, iepzx]
		epyzL = VqL[:, :, iepyz]
		vzL = VqL[:, :, ivz]

		epzxR = VqR[:, :, iepzx]
		epyzR = VqR[:, :, iepyz]
		vzR = VqR[:, :, ivz]

		epzx0 = Vq0[:, :, iepzx]

		epzxNuc = VqNuc[:, :, iepzx]

		# stress/etaS from state minus and plus, (nf, nq)
		theta = (vzR - vzL) + 2.0*csL*(epzxL+epzx0+epzxNuc) + 2.0*csR*(epzxR+epzx0+epzxNuc)

		# Step3: integrate the state variable \Psi1 analytically with constant
		# V0.
		slip0 = 0.02
		exp1 = np.exp(-np.abs(oSR)/slip0 * dt)
		# print("dt", dt)
		SV1 = slip0/np.abs(oSR) * (np.abs(oSR)/slip0 * oSV)**exp1
		# (nf, nq)
		# print("state var 1",SV1[0, 0])
		# print("=====================================")


		# Step4: Compute the slip rate V1 by solving the nonlinear relationship:
		# -1/etaS * (\sigma_n \muF(V1, \Psi1) - \theta) - V1 = 0.
		sigma_n = 40.0
		## Newton update
		def muF(V, SV, x):
			if np.abs(x[0]) < 2.0:
				rs_a = 0.008
			else:
				rs_a = 0.016
			rs_b = 0.012
			sr_0 = 1.0e-6 # reference slip rate
			if np.abs(x[0]) < 0.5:
				muF0 = 0.6
			else:
				muF0 = 0.6
			sl0 = slip0
			c = 0.5/sr_0 * np.exp( \
				(muF0 + rs_b*np.log(sr_0/sl0*SV))/rs_a \
				)

			return rs_a * np.arcsinh(c*V)

		def dmuF(V, SV, x):
			if np.abs(x[0]) < 2.0:
				rs_a = 0.008
			else:
				rs_a = 0.016
			rs_b = 0.012
			sr_0 = 1.0e-6 # reference slip rate
			if np.abs(x[0]) < 0.5:
				muF0 = 0.6
			else:
				muF0 = 0.6
			sl0 = slip0
			c = 0.5/sr_0 * np.exp( \
				(muF0 + rs_b*np.log(sr_0/sl0*SV))/rs_a \
				)

			return rs_a * c / np.sqrt(1.0 + (c*V)**2)

		def func_g(V, SV, absTheta, etaS, sigN, x):
			# print("1st t",absTheta)
			# print("2nd t",1.0/etaS * (sigN*muF(V, SV)))
			# print("3rd t",V)

			# print("muF",muF(V, SV, x))
			# import time
			# time.sleep(0.1)

			return absTheta - 1.0/etaS * (sigN*muF(V, SV, x)) - V

		def func_dg(V, SV, absTheta, etaS, sigN, x):
			# print("2nd t",1.0/etaS * (sigN*dmuF(V, SV)))

			return -1.0/etaS * sigN * dmuF(V, SV, x) - 1.0

		## Newton iteration
		V1 = np.zeros_like(oSR)
		nf = oSR.shape[0]
		nq = oSR.shape[1]
		# Initial guess, V0 = |oSR|
		V0 = np.abs(oSR)

		import time
		start_time = time.perf_counter()

		# ====================================
		# Avoid for loop in the following newton iteration
		# ====================================
		def muFMat(V, SV, xs):
			rs_a = np.zeros_like(V)
			rs_a[:,:] = 0.016
			ind_vw = np.where(np.abs(xs[:,:,0]) < 2.0)
			rs_a[ind_vw] = 0.008

			rs_b = 0.012
			sr_0 = 1.0e-6 # reference slip rate
			muF0 = 0.6
			sl0 = slip0
			c = 0.5/sr_0 * np.exp( \
				(muF0 + rs_b*np.log(sr_0/sl0*SV))/rs_a \
				)

			return rs_a * np.arcsinh(c*V)

		def dmuFMat(V, SV, xs):
			rs_a = np.zeros_like(V)
			rs_a[:,:] = 0.016
			ind_vw = np.where(np.abs(xs[:,:,0]) < 2.0)
			rs_a[ind_vw] = 0.008

			rs_b = 0.012
			sr_0 = 1.0e-6 # reference slip rate
			muF0 = 0.6
			sl0 = slip0
			c = 0.5/sr_0 * np.exp( \
				(muF0 + rs_b*np.log(sr_0/sl0*SV))/rs_a \
				)

			return rs_a * c / np.sqrt(1.0 + (c*V)**2)

		def funcMat_g(V, SV, absTheta, etaS, sigN, xs):

			return absTheta - 1.0/etaS * (sigN*muFMat(V, SV, xs)) - V

		def funcMat_dg(V, SV, absTheta, etaS, sigN, xs):

			return -1.0/etaS * sigN * dmuFMat(V, SV, xs) - 1.0

		for i_iter in range(100):
			x = xs[:, :, :]
			g = funcMat_g(V0[:, :], SV1[:, :], np.abs(theta[:, :]),
					etaS, np.abs(sigma_n),x)
			dg = funcMat_dg(V0[:, :], SV1[:, :], np.abs(theta[:, :]),
							etaS, np.abs(sigma_n),x)
			V1[:, :	] = np.maximum(1e-45, V0[:, :] - g/dg)
			all_converged = np.all(np.abs(g) < 1.0e-10)
			if all_converged:
				print("Converged at i_iter",i_iter,"slip rate 1,g",V1[1, 0],g[1, 0])
				print("=====================================")
				break
			# Update
			V0 = V1.copy()

		# for i_iter in range(100):
		# 	all_converged = True
		# 	for i_f in range(nf):
		# 		for i_q in range(nq):
		# 			x = xs[i_f, i_q, :]
		# 			# Newton iteration for V1, max iteration 100
		# 			# Compute g and dg
		# 			g = func_g(V0[i_f, i_q], SV1[i_f, i_q], np.abs(theta[i_f, i_q]),
		# 					etaS, np.abs(sigma_n),x)
		# 			dg = func_dg(V0[i_f, i_q], SV1[i_f, i_q], np.abs(theta[i_f, i_q]),
		# 					etaS, np.abs(sigma_n),x)
		# 			# Newton update
		# 			V1[i_f, i_q] = max(1e-45, V0[i_f, i_q] - g/dg)
		# 			# Check convergence
		# 			all_converged = all_converged and np.abs(g) < 1.0e-10
		# 			if i_f == 1 and i_q == 0:
		# 				print(i_q)
		# 				print("i_iter",i_iter,"V1,g",V1[i_f, i_q],g)
		# 	# Check convergence
		# 	if all_converged:
		# 		print("Converged at i_iter",i_iter,"slip rate 1,g",V1[1, 0],g)
		# 		print("=====================================")
		# 		break
		# 	# Update
		# 	V0 = V1.copy()

		# End time
		end_time = time.perf_counter()

		# Time spent in seconds
		time_spent = end_time - start_time
		print(f"Time spent in newton method: {time_spent} seconds")
		time.sleep(1)

		V1[:,:] = 0.5*V1[:,:] + 0.5*np.abs(oSR[:,:])
		# Step5: update state variables with 0.5*(V1 + oSR)
		exp1 = np.exp(-np.abs(V1)/slip0 * dt)
		SV2 = slip0/np.abs(V1) * (np.abs(V1)/slip0 * oSV)**exp1
		print("state var 2",SV2[1, 0])
		print("=====================================")


		# Step6: resolve the slip rate with SV2
		## Newton iteration
		V2 = np.zeros_like(V1)
		# Initial guess
		V0 = V1.copy()

		for i_iter in range(100):
			x = xs[:, :, :]
			g = funcMat_g(V0[:, :], SV2[:, :], np.abs(theta[:, :]),
					etaS, np.abs(sigma_n),x)
			dg = funcMat_dg(V0[:, :], SV2[:, :], np.abs(theta[:, :]),
							etaS, np.abs(sigma_n),x)
			V2[:, :	] = np.maximum(1e-45, V0[:, :] - g/dg)
			all_converged = np.all(np.abs(g) < 1.0e-10)
			if all_converged:
				print("Converged at i_iter",i_iter,"slip rate 2,g",V2[1, 0],g[1, 0])
				print("=====================================")
				break
			# Update
			V0 = V2.copy()

		# for i_iter in range(100):
		# 	all_converged = True
		# 	for i_f in range(nf):
		# 		for i_q in range(nq):
		# 			x = xs[i_f, i_q, :]
		# 			# Newton iteration for V1, max iteration 100
		# 			# Compute g and dg
		# 			g = func_g(V0[i_f, i_q], SV2[i_f, i_q], np.abs(theta[i_f, i_q]),
		# 					etaS, np.abs(sigma_n),x)
		# 			dg = func_dg(V0[i_f, i_q], SV2[i_f, i_q], np.abs(theta[i_f, i_q]),
		# 					etaS, np.abs(sigma_n),x)
		# 			# Newton update
		# 			V2[i_f, i_q] = max(1e-45, V0[i_f, i_q] - g/dg)
		# 			# Check convergence
		# 			all_converged = all_converged and np.abs(g) < 1.0e-10
		# 			if i_f == 1 and i_q == 0:
		# 				print("i_iter",i_iter,"V2,g",V2[i_f, i_q],g)
		# 	# Check convergence
		# 	if all_converged:
		# 		print("Converged at i_iter",i_iter,"slip rate 2,g",V2[2, 0],g)
		# 		print("1/eta * shear",1/etaS * (sigma_n*muF(V2[2, 0], SV2[2, 0],
		# 			x = xs[2, 0, :])))
		# 		print("=====================================")
		# 		break
		# 	# Update
		# 	V0 = V2.copy()

		## store the updated state variable and slip rate in stVars and slRates
		stVars[:,:] = SV2[:,:]
		# find index in theta[i_f,i_q], if theta >= 0, V2 = V2, else V2 = -V2
		ind_pos = np.where(theta >= 0.0)
		slRates[ind_pos] = V2[ind_pos]
		ind_neg = np.where(theta < 0.0)
		slRates[ind_neg] = -V2[ind_neg]

		# Step6: Compute b and c of the basic variables on two side
		# from SV2 and V2

		shearTraction = etaS * ( (theta - 2.0*csL*epzx0 - 2.0*csR*epzx0)\
								- slRates)
		ezx_b = shearTraction / muL / 2.0
		ezx_c = shearTraction / muR / 2.0

		v_b = vzL + 2.0*csL*(ezx_b - epzxL)
		v_c = vzR + 2.0*csR*(epzxR - ezx_c)

		# ##=========================================
		# ## compute from continuous conditions, test
		# ##=========================================
		# aS = epzxR - epzxL
		# bS = (vzR - vzL)/csL/2.0

		# alpha1 = 0.5*(aS + bS)
		# alpha3 = 0.5*(aS - bS)

		# ezx_b = epzxL + alpha1
		# # ezx_c = epzxR - alpha3
		# ezx_c = ezx_b

		# v_b = vzL + 2.0*csL*alpha1
		# # v_c = vzR + 2.0*csR*alpha3
		# v_c = v_b

		eyz_b = epyzL
		eyz_c = epyzR

		Vqb = np.zeros_like(UqL)
		Vqc = np.zeros_like(UqR)
		Vqb[:, :, iepzx] = ezx_b
		Vqb[:, :, iepyz] = eyz_b
		Vqb[:, :, ivz] = v_b

		Vqc[:, :, iepzx] = ezx_c
		Vqc[:, :, iepyz] = eyz_c
		Vqc[:, :, ivz] = v_c
		# Step7: Compute surface flux based on b and c.
		AL = np.zeros([ns, ns])
		AL[0, 2] = -0.5
		AL[2, 0] = -2.0 * muL/rhoL
		AR = np.zeros([ns, ns])
		AR[0, 2] = -0.5
		AR[2, 0] = -2.0 * muR/rhoR

		# Compute the flux by multiplying AL with Vqb and R with Vqc
		rFqL = np.zeros_like(UqL)
		rFqR = np.zeros_like(UqR)
		for i_f in range(nf):
			for i_q in range(nq):
				rFqL[i_f, i_q, :] = AL.dot(Vqb[i_f, i_q, :])
				rFqR[i_f, i_q, :] = AR.dot(Vqc[i_f, i_q, :])

		# Step8: Rotate the flux back to the original coordinate system
		FqL, FqR = self.get_rotated_var(rFqL, rFqR, n_hat, s_hat, ns, inv=True)

		# # Normalize the normal vectors
		# n_mag = np.linalg.norm(normals, axis=2, keepdims=True)
		# n_hat = normals/n_mag

		# # Left flux
		# RuFqL,_ = physics.get_conv_flux_projected(UqL, n_hat)

		# # Right flux
		# RuFqR,_ = physics.get_conv_flux_projected(UqR, n_hat)

		# # Jump
		# dUq = UqR - UqL

		# # Calculate max wave speeds at each point
		# a = physics.compute_variable("MaxWaveSpeed", UqL,
		# 		flag_non_physical=True)
		# aR = physics.compute_variable("MaxWaveSpeed", UqR,
		# 		flag_non_physical=True)

		# idx = aR > a
		# a[idx] = aR[idx]

		# print("Upwind flux:",FqL[0])
		# print("Rusanov flux:",(0.5*(RuFqL+RuFqR) - 0.5*a*dUq)[0])
		# print("Upwind flux R:",FqR[0])

		# # Put together
		# return n_mag*(0.5*(RuFqL+RuFqR) - 0.5*a*dUq), n_mag*(0.5*(RuFqL+RuFqR) - 0.5*a*dUq)
		return n_mag*FqL, n_mag*FqR