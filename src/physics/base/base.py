from abc import ABC, abstractmethod
import code
from enum import Enum, auto
import numpy as np
from scipy.optimize import root
import sys

import errors

import numerics.basis.tools as basis_tools
from physics.base.data import ICData, BCData, ExactData, SourceData
import physics.base.functions as base_fcns
from physics.base.functions import BCType as base_BC_type
from physics.base.functions import FcnType as base_fcn_type
from physics.base.functions import ConvNumFluxType as base_conv_num_flux_type


def process_map(fcn_type, fcn_map):
	if fcn_type != "":
		# Update kwargs with reference to desired function 
		for fcn_keys in fcn_map.keys():
			if fcn_keys.name == fcn_type:
				# kwargs.update(Function=fcn_map[fcn_keys])
				fcn_ref = fcn_map[fcn_keys]
				break
	else:
		fcn_ref = None
	return fcn_ref


def set_state_indices_slices(physics):
	# State indices
	physics.StateIndices = {}
	physics.state_slices = {}
	index = 0

	# indices
	for key in physics.StateVariables:
		physics.StateIndices[key.name] = index
		physics.state_slices[key.name] = slice(index, index+1)
		index += 1


class PhysicsBase(object):
	'''
	Class: IFace
	--------------------------------------------------------------------------
	This is a class defined to encapsulate the temperature table with the 
	relevant methods
	'''
	@property
	@abstractmethod
	def NUM_STATE_VARS(self):
		pass

	@property
	@abstractmethod
	def dim(self):
		pass

	@property
	@abstractmethod
	def PHYSICS_TYPE(self):
		# store so this can be easily accessed outside the physics modules
		pass

	def __init__(self, order, basis_type, mesh):
		'''
		Method: __init__
		--------------------------------------------------------------------------
		This method initializes the temperature table. The table uses a
		piecewise linear function for the constant pressure specific heat 
		coefficients. The coefficients are selected to retain the exact 
		enthalpies at the table points.
		'''
		# dim = mesh.dim
		# self.dim = mesh.dim
		self.params = {}
		self.IC = None
		self.exact_soln = None
		self.conv_flux_fcn = None
		self.source_terms = []
		# Boundary conditions
		# self.BCs = []
		# for ibfgrp in range(mesh.num_boundary_groups):
		# 	self.BCs.append(BCData(Name=mesh.BFGNames[ibfgrp]))
		# self.nBC = mesh.num_boundary_groups
		# self.BCs = [BCData() for ibfgrp in range(mesh.num_boundary_groups)]
		# for ibfgrp in range(mesh.num_boundary_groups):
		# 	self.BCs[ibfgrp].Name = mesh.BFGNames[ibfgrp]
		# 	# self.BCs[0].Set(Name=mesh.BFGNames[ibfgrp])
		# self.BCs = [None]*mesh.num_boundary_groups
		self.BCs = dict.fromkeys(mesh.boundary_groups.keys())

		# Basis, Order data for each element group
		# For now, ssume uniform basis and Order for each element group 


		# if type(basis) is str:
		# 	basis = BasisType[basis]
		# self.Basis = basis
		self.order = order
		# if type(order) is int:
		# 	self.order = order
		# elif type(order) is list:
		# 	self.order = order[0]
		# else:
		# 	raise Exception("Input error")

		basis = basis_tools.set_basis(self.order, basis_type)
		self.U = np.zeros([mesh.num_elems, basis.get_num_basis_coeff(
				self.order), self.NUM_STATE_VARS])
		self.S = np.zeros([mesh.num_elems, basis.get_num_basis_coeff(
				self.order), self.NUM_STATE_VARS])
		set_state_indices_slices(self)


		if mesh.dim != self.dim:
			raise errors.IncompatibleError

		self.set_maps()

	def __repr__(self):
		return '{self.__class__.__name__}'.format(self=self)

	def set_maps(self):

		self.IC_fcn_map = {
			base_fcn_type.Uniform : base_fcns.Uniform,
		}

		self.exact_fcn_map = {
			base_fcn_type.Uniform : base_fcns.Uniform,
		}

		self.BC_map = {
			base_BC_type.StateAll : base_fcns.StateAll,
			base_BC_type.Extrapolate : base_fcns.Extrapolate,
		}

		self.BC_fcn_map = {
			base_fcn_type.Uniform : base_fcns.Uniform,
		}

		self.source_map = {}

		self.conv_num_flux_map = {}
		if "MaxWaveSpeed" in self.AdditionalVariables.__members__:
			self.conv_num_flux_map.update({
				base_conv_num_flux_type.LaxFriedrichs : base_fcns.LaxFriedrichs,
			})

	def set_physical_params(self):
		pass

	# def SetParams(self, **kwargs):
	# 	Params = self.params
	# 	# Overwrite
	# 	for key in kwargs:
	# 		Params[key] = kwargs[key]
			# if key not in Params.keys(): raise Exception("Input error")
			# if key is "ConvFluxNumerical":
			# 	Params[key] = self.ConvFluxType[kwargs[key]]
			# else:
			# 	Params[key] = kwargs[key]

	def set_IC(self, IC_type, **kwargs):
		fcn_ref = process_map(IC_type, self.IC_fcn_map)
		self.IC = fcn_ref(**kwargs)

	def set_exact(self, exact_type, **kwargs):
		fcn_ref = process_map(exact_type, self.exact_fcn_map)
		self.exact_soln = fcn_ref(**kwargs)

	def set_BC(self, bname, BC_type, fcn_type=None, **kwargs):
		if self.BCs[bname] is not None:
			raise ValueError
		else:
			if fcn_type is not None:
				fcn_ref = process_map(fcn_type, self.BC_fcn_map)
				kwargs.update(function=fcn_ref)
			BC_ref = process_map(BC_type, self.BC_map)
			BC = BC_ref(**kwargs)
			self.BCs[bname] = BC

		# for i in range(len(self.BCs)):
		# 	BC = self.BCs[i]
		# 	if BC is None:
		# 		if fcn_type is not None:
		# 			fcn_ref = process_map(fcn_type, self.BC_fcn_map)
		# 			kwargs.update(function=fcn_ref)
		# 		BC_ref = process_map(BC_type, self.BC_map)
		# 		BC = BC_ref(**kwargs)
		# 		self.BCs[i] = BC
		# 		break

	# def SetBC(self, BCName, **kwargs):
	# 	found = False
	# 	code.interact(local=locals())
	# 	for BC in self.BCs:
	# 		if BC.Name == BCName:
	# 			BC.Set(**kwargs)
	# 			found = True
	# 			break

	# 	if not found:
	# 		raise NameError

	def set_source(self, source_type, **kwargs):
		source_ref = process_map(source_type, self.source_map)
		source = source_ref(**kwargs)
		self.source_terms.append(source)

	def set_conv_num_flux(self, conv_num_flux_type, **kwargs):
		conv_num_flux_ref = process_map(conv_num_flux_type, 
				self.conv_num_flux_map)
		self.conv_flux_fcn = conv_num_flux_ref(**kwargs)
		
	@abstractmethod
	class StateVariables(Enum):
		pass

	class AdditionalVariables(Enum):
		pass

	def get_state_index(self, var_name):
		# idx = self.VariableType[VariableName]
		idx = self.StateIndices[var_name]
		# idx = self.StateVariables.__members__.keys().index(VariableName)
		return idx

	def get_state_slice(self, var_name):
		# idx = self.VariableType[VariableName]
		slc = self.state_slices[var_name]
		# idx = self.StateVariables.__members__.keys().index(VariableName)
		return slc

	# @abstractmethod
	# class BCType(IntEnum):
	# 	pass

	# @abstractmethod
	# class BCTreatment(IntEnum):
	# 	pass

	# def SetBCTreatment(self):
	# 	# default is Prescribed
	# 	self.BCTreatments = {n:self.BCTreatment.Prescribed for n in range(len(self.BCType))}
	# 	self.BCTreatments[self.BCType.StateAll] = self.BCTreatment.Riemann

	# @abstractmethod
	# class ConvFluxType(IntEnum):
	# 	pass

	# def SetSource(self, **kwargs):
	# 	#append src data to source_terms list 
	# 	Source = SourceData()
	# 	self.source_terms.append(Source)
	# 	Source.Set(**kwargs)

	def get_quadrature_order(self, order):
		return 2*order+1

	@abstractmethod
	def get_conv_flux_interior(self, u):
		pass

	@abstractmethod
	def get_conv_flux_numerical(self, UpL, UpR, normals):
		# self.conv_flux_fcn.AllocHelperArrays(uL)
		F = self.conv_flux_fcn.compute_flux(self, UpL, UpR, normals)

		return F

	# @abstractmethod
	# def BoundaryState(self, BC, nq, xglob, Time, normals, uI):
	# 	pass

	#Source state takes multiple source terms (if needed) and sums them together. 
	def eval_source_terms(self, nq, xglob, Time, Up, s=None):
		for Source in self.source_terms:

			#loop through available source terms
			Source.x = xglob
			Source.nq = nq
			Source.time = Time
			Source.U = Up
			s += self.CallSourceFunction(Source,Source.x,Source.time)

		return s

	def eval_source_term_jacobians(self, nq, xglob, Time, Up, jac=None):
		for Source in self.source_terms:
			#loop through available source terms
			Source.x = xglob
			Source.nq = nq
			Source.time = Time
			Source.U = Up
			jac += self.CallSourceJacobianFunction(Source,Source.x,Source.time)

		return jac
		
	def get_conv_flux_projected(self, Up, normals):

		F = self.get_conv_flux_interior(Up)
		return np.sum(F.transpose(1,0,2)*normals, axis=2).transpose()

	# def ConvFluxBoundary(self, BC, uI, uB, normals, nq, data):
	# 	bctreatment = self.BCTreatments[BC.BCType]
	# 	if bctreatment == self.BCTreatment.Riemann:
	# 		F = self.ConvFluxNumerical(uI, uB, normals, nq, data)
	# 	else:
	# 		# Prescribe analytic flux
	# 		try:
	# 			Fa = data.Fa
	# 		except AttributeError:
	# 			data.Fa = Fa = np.zeros([nq, self.NUM_STATE_VARS, self.dim])
	# 		# Fa = self.get_conv_flux_interior(uB, Fa)
	# 		# # Take dot product with n
	# 		try: 
	# 			F = data.F
	# 		except AttributeError:
	# 			data.F = F = np.zeros_like(uI)
	# 		F[:] = self.get_conv_flux_projected(uB, normals)

	# 	return F

	def compute_variable(self, scalar_name, Uq, flag_non_physical=False):
		# if type(ScalarNames) is list:
		# 	nscalar = len(ScalarNames)
		# elif type(ScalarNames) is str:
		# 	nscalar = 1
		# 	ScalarNames = [ScalarNames]
		# else:
		# 	raise TypeError

		# nq = U.shape[0]
		# if scalar is None or scalar.shape != (nq, nscalar):
		# 	scalar = np.zeros([nq, nscalar])
		# scalar = np.zeros([Up.shape[0], 1])

		# for iscalar in range(nscalar):
		# 	sname = ScalarNames[iscalar]
		# 	try:
		# 		sidx = self.get_state_index(sname)
		# 		scalar[:,iscalar] = U[:,sidx]
		# 	# if sidx < self.NUM_STATE_VARS:
		# 	# 	# State variable
		# 	# 	scalar[:,iscalar] = U[:,sidx]
		# 	# else:
		# 	except KeyError:
		# 		scalar[:,iscalar:iscalar+1] = self.compute_additional_variable(sname, U, scalar[:,iscalar:iscalar+1],
		# 			flag_non_physical)

		try:
			sidx = self.get_state_index(scalar_name)
			# scalar[:,iscalar] = Up[:,sidx]
			scalar = Uq[:, sidx:sidx+1].copy()
		# if sidx < self.NUM_STATE_VARS:
		# 	# State variable
		# 	scalar[:,iscalar] = U[:,sidx]
		# else:
		except KeyError:
			scalar = self.compute_additional_variable(scalar_name, Uq, 
					flag_non_physical)

		return scalar

	def compute_additional_variable(self, ScalarName, Up, flag_non_physical):
		pass

	def call_function(self, FcnData, x, t):
		# for key in kwargs:
		# 	if key is "x":
		# 		FcnData.x = kwargs[key]
		# 		FcnData.nq = FcnData.x.shape[0]
		# 	elif key is "Time":
		# 		FcnData.time = kwargs[key]
		# 	else:
		# 		raise Exception("Input error")

		# nq = FcnData.nq
		# sr = self.NUM_STATE_VARS
		# if FcnData.U is None or FcnData.U.shape != (nq, sr):
		# 	FcnData.U = np.zeros([nq, sr], dtype=self.U.dtype)

		# FcnData.U[:] = FcnData.Function(self, FcnData)
		# FcnData.alloc_helpers([x.shape[0], self.NUM_STATE_VARS])
		FcnData.Up = FcnData.get_state(self, x, t)

		return FcnData.Up

	def CallSourceFunction(self, FcnData, x, t):
		# for key in kwargs:
		# 	if key is "x":
		# 		FcnData.x = kwargs[key]
		# 		FcnData.nq = FcnData.x.shape[0]
		# 	elif key is "Time":
		# 		FcnData.time = kwargs[key]
		# 	else:
		# 		raise Exception("Input error")

		# nq = FcnData.nq
		# sr = self.NUM_STATE_VARS
		# if FcnData.S is None or FcnData.S.shape != (nq, sr):
		# 	FcnData.S = np.zeros([nq, sr], dtype=self.S.dtype)
		# code.interact(local=locals())
		FcnData.S = FcnData.get_source(self, FcnData, x, t)

		return FcnData.S

	def CallSourceJacobianFunction(self, FcnData, x, t):
		# for key in kwargs:
		# 	if key is "x":
		# 		FcnData.x = kwargs[key]
		# 		FcnData.nq = FcnData.x.shape[0]
		# 	elif key is "Time":
		# 		FcnData.time = kwargs[key]
		# 	else:
		# 		raise Exception("Input error")

		# nq = FcnData.nq
		# sr = self.NUM_STATE_VARS
		# if FcnData.S is None or FcnData.S.shape != (nq, sr):
		# 	FcnData.S = np.zeros([nq, sr], dtype=self.S.dtype)
		# code.interact(local=locals())
		FcnData.jac = FcnData.get_jacobian(self, FcnData, x, t)
		return FcnData.jac

	# def FcnUniform(self, FcnData):
	# 	Data = FcnData.Data
	# 	U = FcnData.U
	# 	ns = self.NUM_STATE_VARS

	# 	for k in range(ns):
	# 		U[:,k] = Data.State[k]

	# 	return U