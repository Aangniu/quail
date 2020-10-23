# ------------------------------------------------------------------------ #
#
#       File : src/numerics/limiting/positivitypreserving.py
#
#       Contains class definitions for positivity-preserving limiters.
#
# ------------------------------------------------------------------------ #
from abc import ABC, abstractmethod
import numpy as np

import errors
import general

import meshing.tools as mesh_tools

import numerics.helpers.helpers as helpers
import numerics.limiting.base as base


POS_TOL = 1.e-10


class PositivityPreserving(base.LimiterBase):
	'''
	This class corresponds to the positivity-preserving limiter for the
	Euler equations. It inherits from the LimiterBase class. See
	See LimiterBase for detailed comments of attributes and methods. See
	the following references:
		[1] X. Zhang, C.-W. Shu, "On positivity-preserving high order
		discontinuous Galerkin schemes for compressible Euler equations
		on rectangular meshes," Journal of Computational Physics.
		229:8918–8934, 2010.
		[2] C. Wang, X. Zhang, C.-W. Shu, J. Ning, "Robust high order
		discontinuous Galerkin schemes for two-dimensional gaseous
		detonations," Journal of Computational Physics, 231:653-665, 2012.

	Attributes:
	-----------
	var_name1: str
		name of first variable involved in limiting (density)
	var_name2: str
		name of second variable involved in limiting (pressure)
	elem_vols: numpy array
		element volumes
	basis_val_elem_faces: numpy array
		stores basis values for element and faces
	quad_wts_elem: numpy array
		quadrature points for element
	djac_elems: numpy array
		stores Jacobian determinants for each element
	'''
	COMPATIBLE_PHYSICS_TYPES = general.PhysicsType.Euler

	def __init__(self, physics_type):
		super().__init__(physics_type)
		self.var_name1 = "Density"
		self.var_name2 = "Pressure"
		self.elem_vols = np.zeros(0)
		self.basis_val_elem_faces = np.zeros(0)
		self.quad_wts_elem = np.zeros(0)
		self.djac_elems = np.zeros(0)

	def precompute_helpers(self, solver):
		# Unpack
		elem_helpers = solver.elem_helpers
		int_face_helpers = solver.int_face_helpers
		self.elem_vols, _ = mesh_tools.element_volumes(solver.mesh, solver)

		# Basis values in element interior and on faces
		if not solver.basis.skip_interp:
			basis_val_faces = int_face_helpers.faces_to_basisL.copy()
			bshape = basis_val_faces.shape
			basis_val_faces.shape = (bshape[0]*bshape[1], bshape[2])
			self.basis_val_elem_faces = np.vstack((elem_helpers.basis_val,
					basis_val_faces))
		else:
			self.basis_val_elem_faces = elem_helpers.basis_val

		# Jacobian determinant
		self.djac_elems = elem_helpers.djac_elems

		# Element quadrature weights
		self.quad_wts_elem = elem_helpers.quad_wts

	def limit_element(self, solver, Uc):
		# Unpack
		physics = solver.physics
		elem_helpers = solver.elem_helpers
		int_face_helpers = solver.int_face_helpers

		djac = self.djac_elems

		# Interpolate state at quadrature points over element and on faces
		U_elem_faces = helpers.evaluate_state(Uc, self.basis_val_elem_faces,
				skip_interp=solver.basis.skip_interp)
		nq_elem = self.quad_wts_elem.shape[0]
		U_elem = U_elem_faces[:, :nq_elem, :]

		# Average value of state
		U_bar = helpers.get_element_mean(U_elem, self.quad_wts_elem, djac,
				self.elem_vols)

		# Density and pressure from averaged state
		rho_bar = physics.compute_variable(self.var_name1, U_bar)
		p_bar = physics.compute_variable(self.var_name2, U_bar)

		if np.any(rho_bar < 0.) or np.any(p_bar < 0.):
			raise errors.NotPhysicalError

		# Ignore divide-by-zero
		np.seterr(divide='ignore')

		''' Limit density '''
		# Compute density at quadrature points
		rho_elem_faces = physics.compute_variable(self.var_name1,
				U_elem_faces)
		# Check if limiting is needed
		theta = np.abs((rho_bar - POS_TOL)/(rho_bar - rho_elem_faces))
		theta1 = np.empty(theta.shape[0])
		for elem_ID in range(theta.shape[0]):
			theta1[elem_ID] = np.amin([1., np.amin(theta[elem_ID])])

			# Rescale
			if theta1[elem_ID] < 1.:
				irho = physics.get_state_index(self.var_name1)
				Uc[elem_ID, :, irho] = theta1[elem_ID]*Uc[elem_ID, :, irho] + (1. -
						theta1[elem_ID])*rho_bar[elem_ID]

		if np.any(theta1 < 1.):
			# Intermediate limited solution
			U_elem_faces = helpers.evaluate_state(Uc,
					self.basis_val_elem_faces,
					skip_interp=solver.basis.skip_interp)

		''' Limit pressure '''
		# Compute pressure at quadrature points
		p_elem_faces = physics.compute_variable(self.var_name2, U_elem_faces)
		theta[:] = 1.
		for elem_ID in range(theta.shape[0]):
			i_pos_p = (p_elem_faces[elem_ID] < 0.).reshape(-1)
			# Indices where pressure is negative
			theta[elem_ID, i_pos_p] = p_bar[elem_ID]/(p_bar[elem_ID] - p_elem_faces[elem_ID, i_pos_p])
			theta2 = np.amin(theta[elem_ID])
			# Rescale
			if theta2 < 1.:
				Uc[elem_ID] = theta2*Uc[elem_ID] + (1. - theta2)*U_bar[elem_ID]

		np.seterr(divide='warn')

		return Uc


class PositivityPreservingChem(PositivityPreserving):
	'''
    Class: PPLimiter
    ------------------
    This class contains information about the positivity preserving limiter
    '''

	COMPATIBLE_PHYSICS_TYPES = general.PhysicsType.Chemistry

	def __init__(self, physics_type):
		'''
		Method: __init__
		-------------------
		Initializes PPLimiter object
		'''
		super().__init__(physics_type)
		# self.var_name1 = "Density"
		# self.var_name2 = "Pressure"
		self.scalar3 = "Mixture"
		# self.elem_vols = np.zeros(0)
		# self.basis_val_elem_faces = None
		# self.quad_wts_elem = np.zeros(0)
		# self.djac_elems = np.zeros(0)

	# def precompute_helpers(self, solver):
	# 	elem_helpers = solver.elem_helpers
	# 	int_face_helpers = solver.int_face_helpers
	# 	_, self.elem_vols = mesh_tools.element_volumes(solver.mesh, solver)

	# 	# basis values in element interior and on faces
	# 	basis_val_faces = int_face_helpers.faces_to_basisL.copy()
	# 	bshape = basis_val_faces.shape
	# 	basis_val_faces.shape = (bshape[0]*bshape[1], bshape[2])
	# 	self.basis_val_elem_faces = np.vstack((elem_helpers.basis_val, basis_val_faces))

	# 	# Jacobian determinant
	# 	self.djac_elems = elem_helpers.djac_elems

	# 	# Element quadrature weights
	# 	self.quad_wts_elem = elem_helpers.quad_wts

	def limit_element(self, solver, elem, U):
		'''
		Method: limit_element
		------------------------
		Limits the solution on each element

		Inputs:
			solver: type of solver (e.g., DG, ADER-DG, etc...)
			elem: element index

		Outputs:
			U: solution array
		'''
		physics = solver.physics
		elem_helpers = solver.elem_helpers
		int_face_helpers = solver.int_face_helpers

		djac = self.djac_elems[elem]

		# interpolate state and gradient at quadrature points over element
		# and on faces
		# u_elem_faces = np.matmul(self.basis_val_elem_faces, U)
		u_elem_faces = helpers.evaluate_state(U, self.basis_val_elem_faces,
				skip_interp=solver.basis.skip_interp)
		nq_elem = self.quad_wts_elem.shape[0]
		u_elem = u_elem_faces[:nq_elem,:]

		# Average value of state
		vol = self.elem_vols[elem]
		# u_bar = np.matmul(u_elem.transpose(), self.quad_wts_elem*djac).T/vol
		u_bar = helpers.get_element_mean(u_elem, self.quad_wts_elem, djac,
				self.elem_vols[elem])

		# Density and pressure
		rho_bar = physics.compute_variable(self.var_name1, u_bar)
		p_bar = physics.compute_variable(self.var_name2, u_bar)
		rhoY_bar = physics.compute_variable(self.scalar3, u_bar)

		if np.any(rho_bar < 0.) or np.any(p_bar < 0.) or np.any(rhoY_bar < 0.):
			raise errors.NotPhysicalError

		np.seterr(divide='ignore')

		''' Limit density '''
		# Compute density
		rho_elem_faces = physics.compute_variable(self.var_name1, u_elem_faces)
		theta = np.abs((rho_bar - POS_TOL)/(rho_bar - rho_elem_faces))
		theta1 = np.amin([1., np.amin(theta)])

		# Rescale
		if theta1 < 1.:
			irho = physics.get_state_index(self.var_name1)
			U[:,irho] = theta1*U[:,irho] + (1. - theta1)*rho_bar

			# Intermediate limited solution
			# u_elem_faces = np.matmul(self.basis_val_elem_faces, U)
			u_elem_faces = helpers.evaluate_state(U, self.basis_val_elem_faces,
					skip_interp=solver.basis.skip_interp)

		''' Limit mass fraction '''
		rhoY_elem_faces = physics.compute_variable(self.scalar3, u_elem_faces)
		theta = np.abs(rhoY_bar/(rhoY_bar-rhoY_elem_faces+POS_TOL))
		theta2 = np.amin([1.,np.amin(theta)])

		# Rescale
		if theta2 < 1.:
			irhoY = physics.get_state_index(self.scalar3)
			U[:,irhoY] = theta2*U[:,irhoY] + (1. - theta2)*rhoY_bar

			# Intermediate limited solution
			# u_elem_faces = np.matmul(self.basis_val_elem_faces, U)
			u_elem_faces = helpers.evaluate_state(U, self.basis_val_elem_faces,
					skip_interp=solver.basis.skip_interp)
		# code.interact(local=locals())


		''' Limit pressure '''
		p_elem_faces = physics.compute_variable(self.var_name2, u_elem_faces)
		# theta = np.abs((p_bar - POS_TOL)/(p_bar - p_elem_faces))
		theta[:] = 1.
		i_pos_p = (p_elem_faces < 0.).reshape(-1) # indices where pressure is negative
		theta[i_pos_p] = p_bar/(p_bar - p_elem_faces[i_pos_p])
		thetax = np.amin(theta)
		if thetax < 1.:
			U = thetax*U + (1. - thetax)*u_bar

		np.seterr(divide='warn')

		return U
