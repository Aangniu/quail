import numpy as np 
from Basis import *
from Quadrature import *
from Mesh import *
import code
import copy
from Data import ArrayList, GenericData
from SolverTools import *
from Stepper import *
import time
import MeshTools
import Post
import Errors
from scipy.optimize import root
import Limiter

global echeck
echeck = -1

class DG_Solver(object):
	'''
	Class: DG_Solver
	--------------------------------------------------------------------------
	Discontinuous Galerkin method designed to solve a given set of PDEs

	ATTRIBUTES:
		Params: list of parameters for the solver
		EqnSet: solver object (current implementation supports Scalar and Euler equations)
		mesh: mesh object
		DataSet: location to store generic data
		Time: current time in the simulation
		nTimeStep: number of time steps
	'''
	def __init__(self,Params,EqnSet,mesh):
		'''
		Method: __init__
		--------------------------------------------------------------------------
		Initializes the DG_Solver object, verifies parameters, and initializes the state

		INPUTS:
			Params: list of parameters for the solver
			EqnSet: solver object (current implementation supports Scalar and Euler equations)
			mesh: mesh object
		'''

		self.Params = Params
		self.EqnSet = EqnSet
		self.mesh = mesh
		self.DataSet = GenericData()

		self.Time = Params["StartTime"]
		self.nTimeStep = 0 # will be set later

		TimeScheme = Params["TimeScheme"]
		if TimeScheme is "FE":
			Stepper = FE()
		elif TimeScheme is "RK4":
			Stepper = RK4()
		elif TimeScheme is "LSRK4":
			Stepper = LSRK4()
		elif TimeScheme is "SSPRK3":
			Stepper = SSPRK3()
		elif TimeScheme is "ADER":
			Stepper = ADER()
		else:
			raise NotImplementedError("Time scheme not supported")
		# if Params["nTimeStep"] > 0:
		# 	Stepper.dt = Params["EndTime"]/Params["nTimeStep"]
		self.Stepper = Stepper

		# Limiter
		limiterType = Params["ApplyLimiter"]
		self.Limiter = Limiter.set_limiter(limiterType)

		# Check validity of parameters
		self.check_solver_params()

		# Initialize state
		self.init_state()


	def check_solver_params(self):
		'''
		Method: check_solver_params
		--------------------------------------------------------------------------
		Checks the validity of the solver parameters

		'''
		Params = self.Params
		mesh = self.mesh
		EqnSet = self.EqnSet
		### Check interp basis validity
		if BasisType[Params["InterpBasis"]] == BasisType.LagrangeSeg or BasisType[Params["InterpBasis"]] == BasisType.LegendreSeg:
		    if mesh.Dim != 1:
		        raise Errors.IncompatibleError
		else:
		    if mesh.Dim != 2:
		        raise Errors.IncompatibleError

		### Check uniform mesh
		if Params["UniformMesh"] is True:
		    ''' 
		    Check that element volumes are the same
		    Note that this is a necessary but not sufficient requirement
		    '''
		    TotVol, ElemVols = MeshTools.element_volumes(mesh)
		    if (ElemVols.Max - ElemVols.Min)/TotVol > 1.e-8:
		        raise ValueError

		### Check linear geometric mapping
		if Params["LinearGeomMapping"] is True:
			if mesh.QOrder != 1:
			    raise Errors.IncompatibleError
			if mesh.QBasis == BasisType.LagrangeQuad \
			    and Params["UniformMesh"] is False:
			    raise Errors.IncompatibleError

		### Check limiter ###
		if Params["ApplyLimiter"] is 'ScalarPositivityPreserving' \
			and EqnSet.StateRank > 1:
				raise IncompatibleError
		if Params["ApplyLimiter"] is 'PositivityPreserving' \
			and EqnSet.StateRank == 1:
				raise IncompatibleError

		### Check time integration scheme ###
		TimeScheme = Params["TimeScheme"]		
		if TimeScheme is "ADER":
			raise Errors.IncompatibleError


	def init_state(self):
		'''
		Method: init_state
		-------------------------
		Initializes the state based on prescribed initial conditions
		
		'''
		
		mesh = self.mesh
		EqnSet = self.EqnSet
		U = EqnSet.U
		ns = EqnSet.StateRank
		Params = self.Params

		# Get mass matrices
		try:
			iMM = self.DataSet.iMM_all
		except AttributeError:
			# not found; need to compute
			iMM_all = get_inv_mass_matrices(mesh, EqnSet, solver=self)

		InterpolateIC = Params["InterpolateIC"]
		quadData = None
		JData = JacobianData(mesh)
		GeomPhiData = None
		quad_pts = None
		xphys = None

		basis = EqnSet.Basis
		order = EqnSet.Order
		rhs = np.zeros([order_to_num_basis_coeff(basis,order),ns],dtype=U.dtype)

		# Precompute basis and quadrature
		if not InterpolateIC:
			QuadOrder,_ = get_gaussian_quadrature_elem(mesh, EqnSet.Basis,
				2*np.amax([order,1]), EqnSet, quadData)

			quadData = QuadData(mesh, mesh.QBasis, EntityType.Element, QuadOrder)

			quad_pts = quadData.quad_pts
			quad_wts = quadData.quad_wts
			nq = quad_pts.shape[0]

			PhiData = BasisData(basis,order,mesh)
			PhiData.eval_basis(quad_pts, Get_Phi=True)
			xphys = np.zeros([nq, mesh.Dim])
		
		else:
			quad_pts, nq = equidistant_nodes(basis, order, quad_pts)
			#nb = nq

		for elem in range(mesh.nElem):

			xphys, GeomPhiData = ref_to_phys(mesh, elem, GeomPhiData, quad_pts, xphys)

			f = EqnSet.CallFunction(EqnSet.IC, x=xphys, Time=self.Time)
			f.shape = nq,ns

			if not InterpolateIC:

				JData.element_jacobian(mesh,elem,quad_pts,get_djac=True)

				iMM = iMM_all[elem]

				#nn = PhiData.nn

				# rhs *= 0.
				# for n in range(nn):
				# 	for iq in range(nq):
				# 		rhs[n,:] += f[iq,:]*PhiData.Phi[iq,n]*quad_wts[iq]*JData.djac[iq*(JData.nq != 1)]
				rhs[:] = np.matmul(PhiData.Phi.transpose(), f*quad_wts*JData.djac) # [nb, ns]

				U[elem,:,:] = np.matmul(iMM,rhs)
			else:
				U[elem] = f

	def apply_limiter(self, U):
		'''
		Method: apply_limiter
		-------------------------
		Applies the limiter to the solution array, U.
		
		INPUTS:
			U: solution array

		OUTPUTS:
			U: limited solution array

		Notes: See Limiter.py for details
		'''
		if self.Limiter is not None:
			self.Limiter.limit_solution(self, U)


	def calculate_residual_elem(self, elem, U, ER, StaticData):
		'''
		Method: calculate_residual_elem
		---------------------------------
		Calculates the volume integral for a specified element
		
		INPUTS:
			elem: element index
			U: solution array
			
		OUTPUTS:
			ER: calculated residiual array (for volume integral of specified element)
		'''
		mesh = self.mesh
		EqnSet = self.EqnSet

		basis = EqnSet.Basis
		order = EqnSet.Order
		entity = EntityType.Element
		ns = EqnSet.StateRank
		dim = EqnSet.Dim

		if Order == 0:
			return ER, StaticData

		if StaticData is None:
			pnq = -1
			quadData = None
			PhiData = None
			JData = JData = JacobianData(mesh)
			xglob = None
			u = None
			F = None
			s = None
			NData = None
			GeomPhiData = None
			StaticData = GenericData()
		else:
			nq = StaticData.pnq
			quadData = StaticData.quadData
			PhiData = StaticData.PhiData
			JData = StaticData.JData
			xglob = StaticData.xglob
			u = StaticData.u
			F = StaticData.F
			s = StaticData.s
			NData = StaticData.NData
			GeomPhiData = StaticData.GeomPhiData


		QuadOrder,QuadChanged = get_gaussian_quadrature_elem(mesh, mesh.QBasis, order, EqnSet, quadData)
		if QuadChanged:
			quadData = QuadData(mesh, basis, entity, QuadOrder)

		quad_pts = quadData.quad_pts
		quad_wts = quadData.quad_wts
		nq = quad_pts.shape[0]

		if QuadChanged:
			PhiData = BasisData(EqnSet.Basis,order,mesh)
			PhiData.eval_basis(quad_pts, Get_Phi=True, Get_GPhi=True)

			xglob = np.zeros([nq, dim])
			u = np.zeros([nq, ns])
			F = np.zeros([nq, ns, dim])
			s = np.zeros([nq, ns])



		JData.element_jacobian(mesh,elem,quad_pts,get_djac=True,get_jac=False,get_ijac=True)
		PhiData.eval_basis(quad_pts, Get_gPhi=True, JData=JData) # gPhi is [nq,nn,dim]

		xglob, GeomPhiData = ref_to_phys(mesh, elem, GeomPhiData, quad_pts, xglob, QuadChanged)

		#nb = PhiData.nb

		# interpolate state and gradient at quad points
		u[:] = np.matmul(PhiData.Phi, U)

		F = EqnSet.ConvFluxInterior(u, F) # [nq,ns,dim]

		# for ir in range(ns):
		# 	for jn in range(nb):
		# 		for iq in range(nq):
		# 			gPhi = PhiData.gPhi[iq,jn] # dim
		# 			ER[jn,ir] += np.dot(gPhi, F[iq,ir,:])*wq[iq]*JData.djac[iq*(JData.nq!=1)]
		ER[:] += np.tensordot(PhiData.gPhi, F*(quad_wts*JData.djac).reshape(nq,1,1), axes=([0,2],[0,2])) # [nb, ns]

		s = np.zeros([nq,ns]) # source is cumulative so it needs to be initialized to zero for each time step
		s = EqnSet.SourceState(nq, xglob, self.Time, NData, u, s) # [nq,ns]
		# Calculate source term integral
		# for ir in range(sr):
		# 	for jn in range(nn):
		# 		for iq in range(nq):
		# 			Phi = PhiData.Phi[iq,jn]
		# 			ER[jn,ir] += Phi*s[iq,ir]*wq[iq]*JData.djac[iq*(JData.nq!=1)]
		ER[:] += np.matmul(PhiData.Phi.transpose(), s*quad_wts*JData.djac) # [nb, ns]
		if elem == echeck:
			code.interact(local=locals())

		# Store in StaticData
		StaticData.pnq = nq
		StaticData.quadData = quadData
		StaticData.PhiData = PhiData
		StaticData.JData = JData
		StaticData.xglob = xglob
		StaticData.u = u
		StaticData.F = F
		StaticData.s = s
		StaticData.NData = NData
		StaticData.GeomPhiData = GeomPhiData

		return ER, StaticData


	def calculate_residual_elems(self, U, R):
		'''
		Method: calculate_residual_elems
		---------------------------------
		Calculates the volume integral across the entire domain
		
		INPUTS:
			U: solution array
			
		OUTPUTS:
			R: calculated residiual array
		'''
		mesh = self.mesh
		EqnSet = self.EqnSet
		StaticData = None

		for elem in range(mesh.nElem):
			R[elem], StaticData = self.calculate_residual_elem(elem, U[elem], R[elem], StaticData)


	def calculate_residual_iface(self, iiface, UL, UR, RL, RR, StaticData):
		'''
		Method: calculate_residual_iface
		---------------------------------
		Calculates the boundary integral for the internal faces
		
		INPUTS:
			iiface: internal face index
			UL: solution array from left neighboring element
			UR: solution array from right neighboring element
			
		OUTPUTS:
			RL: calculated residual array (left neighboring element contribution)
			RR: calculated residual array (right neighboring element contribution)
		'''
		mesh = self.mesh
		dim = mesh.Dim
		EqnSet = self.EqnSet
		ns = EqnSet.StateRank

		IFace = mesh.IFaces[iiface]
		elemL = IFace.ElemL
		elemR = IFace.ElemR
		faceL = IFace.faceL
		faceR = IFace.faceR
		order = EqnSet.Order
		nFacePerElem = mesh.nFacePerElem

		if StaticData is None:
			pnq = -1
			quadData = None
			PhiDataL = None
			PhiDataR = None
			xelemL = None
			xelemR = None
			uL = None
			uR = None
			StaticData = GenericData()
			NData = None
		else:
			nq = StaticData.pnq
			quadData = StaticData.quadData
			PhiDataL = StaticData.PhiDataL
			PhiDataR = StaticData.PhiDataR
			xelemL = StaticData.xelemL
			xelemR = StaticData.xelemR
			uL = StaticData.uL
			uR = StaticData.uR
			NData = StaticData.NData
			Faces2PhiDataL = StaticData.Faces2PhiDataL
			Faces2PhiDataR = StaticData.Faces2PhiDataR

		QuadOrder, QuadChanged = get_gaussian_quadrature_face(mesh, IFace, mesh.QBasis, order, EqnSet, quadData)

		if QuadChanged:
			quadData = QuadData(mesh, EqnSet.Basis, EntityType.IFace, QuadOrder)

		quad_pts = quadData.quad_pts
		quad_wts = quadData.quad_wts
		nq = quad_pts.shape[0]

		if QuadChanged:
			Faces2PhiDataL = [None for i in range(nFacePerElem)]
			Faces2PhiDataR = [None for i in range(nFacePerElem)]

			xelemL = np.zeros([nq, dim])
			xelemR = np.zeros([nq, dim])
			uL = np.zeros([nq, ns])
			uR = np.zeros([nq, ns])

		PhiDataL = Faces2PhiDataL[faceL]
		PhiDataR = Faces2PhiDataR[faceR]

		if PhiDataL is None or QuadChanged:
			Faces2PhiDataL[faceL] = PhiDataL = BasisData(EqnSet.Basis,order,mesh)
			xelemL = PhiDataL.eval_basis_on_face(mesh, faceL, quad_pts, xelemL, Get_Phi=True)
		if PhiDataR is None or QuadChanged:
			Faces2PhiDataR[faceR] = PhiDataR = BasisData(EqnSet.Basis,order,mesh)
			xelemR = PhiDataR.eval_basis_on_face(mesh, faceR, quad_pts[::-1], xelemR, Get_Phi=True)

		NData = iface_normal(mesh, IFace, quad_pts, NData)
		# NData.nvec *= quad_wts

		#nL = PhiDataL.nb
		#nR = PhiDataR.nb
		#nb = np.amax([nL,nR])

		# interpolate state and gradient at quad points
		# for ir in range(ns):
		# 	uL[:,ir] = np.matmul(PhiDataL.Phi, UL[:,ir])
		# 	uR[:,ir] = np.matmul(PhiDataR.Phi, UR[:,ir])
		uL[:] = np.matmul(PhiDataL.Phi, UL)
		uR[:] = np.matmul(PhiDataR.Phi, UR)

		F = EqnSet.ConvFluxNumerical(uL, uR, NData, nq, StaticData) # [nq,ns]

		RL -= np.matmul(PhiDataL.Phi.transpose(), F*quad_wts) # [nb,sr]
		RR += np.matmul(PhiDataR.Phi.transpose(), F*quad_wts) # [nb,sr]

		if elemL == echeck or elemR == echeck:
			if elemL == echeck: print("Left!")
			else: print("Right!")
			code.interact(local=locals())

		# Store in StaticData
		StaticData.pnq = nq
		StaticData.quadData = quadData
		StaticData.PhiDataL = PhiDataL
		StaticData.PhiDataR = PhiDataR
		StaticData.xelemL = xelemL
		StaticData.xelemR = xelemR
		StaticData.uL = uL
		StaticData.uR = uR
		StaticData.NData = NData
		StaticData.Faces2PhiDataL = Faces2PhiDataL
		StaticData.Faces2PhiDataR = Faces2PhiDataR

		return RL, RR, StaticData



	def calculate_residual_ifaces(self, U, R):
		'''
		Method: calculate_residual_ifaces
		-----------------------------------
		Calculates the boundary integral for all internal faces
		
		INPUTS:
			U: solution array
			
		OUTPUTS:
			R: calculated residual array (includes all face contributions)
		'''
		mesh = self.mesh
		EqnSet = self.EqnSet
		StaticData = None

		for iiface in range(mesh.nIFace):
			IFace = mesh.IFaces[iiface]
			elemL = IFace.ElemL
			elemR = IFace.ElemR
			faceL = IFace.faceL
			faceR = IFace.faceR

			UL = U[elemL]
			UR = U[elemR]
			RL = R[elemL]
			RR = R[elemR]

			RL, RR, StaticData = self.calculate_residual_iface(iiface, UL, UR, RL, RR, StaticData)

	def calculate_residual_bface(self, ibfgrp, ibface, U, R, StaticData):
		'''
		Method: calculate_residual_bface
		---------------------------------
		Calculates the boundary integral for the boundary faces
		
		INPUTS:
			ibfgrp: index of boundary face groups (groups indicate different boundary conditions)
			ibface: boundary face index
			U: solution array from internal cell
			
		OUTPUTS:
			R: calculated residual array from boundary face
		'''
		mesh = self.mesh
		EqnSet = self.EqnSet
		ns = EqnSet.StateRank
		dim = mesh.Dim

		BFG = mesh.BFaceGroups[ibfgrp]
		BFace = BFG.BFaces[ibface]
		elem = BFace.Elem
		face = BFace.face
		order = EqnSet.Order
		nFacePerElem = mesh.nFacePerElem

		if StaticData is None:
			pnq = -1
			quadData = None
			PhiData = None
			xglob = None
			uI = None
			uB = None
			NData = None
			GeomPhiData = None
			StaticData = GenericData()
			Faces2xelem = None
		else:
			nq = StaticData.pnq
			quadData = StaticData.quadData
			PhiData = StaticData.PhiData
			xglob = StaticData.xglob
			uI = StaticData.uI
			uB = StaticData.uB
			NData = StaticData.NData
			GeomPhiData = StaticData.GeomPhiData
			Faces2PhiData = StaticData.Faces2PhiData
			Faces2xelem = StaticData.Faces2xelem

		QuadOrder, QuadChanged = get_gaussian_quadrature_face(mesh, BFace, mesh.QBasis, order, EqnSet, quadData)

		if QuadChanged:
			quadData = QuadData(mesh, EqnSet.Basis, EntityType.BFace, QuadOrder)

		quad_pts = quadData.quad_pts
		quad_wts = quadData.quad_wts
		nq = quad_pts.shape[0]

		if QuadChanged:
			Faces2PhiData = [None for i in range(nFacePerElem)]
			# PhiData = BasisData(EqnSet.Basis,Order,nq,mesh)
			xglob = np.zeros([nq, dim])
			uI = np.zeros([nq, ns])
			uB = np.zeros([nq, ns])
			Faces2xelem = np.zeros([nFacePerElem, nq, dim])

		PhiData = Faces2PhiData[face]
		xelem = Faces2xelem[face]
		if PhiData is None or QuadChanged:
			Faces2PhiData[face] = PhiData = BasisData(EqnSet.Basis,order,mesh)
			Faces2xelem[face] = xelem = PhiData.eval_basis_on_face(mesh, face, quad_pts, xelem, Get_Phi=True)

		NData = bface_normal(mesh, BFace, quad_pts, NData)

		PointsChanged = QuadChanged or face != GeomPhiData.face
		xglob, GeomPhiData = ref_to_phys(mesh, elem, GeomPhiData, xelem, xglob, PointsChanged)

		#nn = PhiData.nn

		# interpolate state and gradient at quad points
		# for ir in range(sr):
		# 	uI[:,ir] = np.matmul(PhiData.Phi, U[:,ir])
		uI[:] = np.matmul(PhiData.Phi, U)

		# Get boundary state
		BC = EqnSet.BCs[ibfgrp]
		uB = EqnSet.BoundaryState(BC, nq, xglob, self.Time, NData, uI, uB)

		# NData.nvec *= wq
		F = EqnSet.ConvFluxBoundary(BC, uI, uB, NData, nq, StaticData) # [nq,sr]

		R -= np.matmul(PhiData.Phi.transpose(), F*quad_wts) # [nn,sr]

		if elem == echeck:
			code.interact(local=locals())

		# Store in StaticData
		StaticData.pnq = nq
		StaticData.quadData = quadData
		StaticData.PhiData = PhiData
		StaticData.uI = uI
		StaticData.uB = uB
		StaticData.F = F
		StaticData.xglob = xglob
		StaticData.NData = NData
		StaticData.GeomPhiData = GeomPhiData
		StaticData.Faces2PhiData = Faces2PhiData
		StaticData.Faces2xelem = Faces2xelem

		return R, StaticData


	def calculate_residual_bfaces(self, U, R):
		'''
		Method: calculate_residual_bfaces
		-----------------------------------
		Calculates the boundary integral for all the boundary faces
		
		INPUTS:
			U: solution array from internal cell
			
		OUTPUTS:
			R: calculated residual array from boundary face
		'''
		mesh = self.mesh
		EqnSet = self.EqnSet
		StaticData = None

		for ibfgrp in range(mesh.nBFaceGroup):
			BFG = mesh.BFaceGroups[ibfgrp]

			for ibface in range(BFG.nBFace):
				BFace = BFG.BFaces[ibface]
				elem = BFace.Elem
				face = BFace.face

				R[elem], StaticData = self.calculate_residual_bface(ibfgrp, ibface, U[elem], R[elem], StaticData)


	def calculate_residual(self, U, R):
		'''
		Method: calculate_residual
		-----------------------------------
		Calculates the boundary + volume integral for the DG formulation
		
		INPUTS:
			U: solution array
			
		OUTPUTS:
			R: residual array
		'''

		mesh = self.mesh
		EqnSet = self.EqnSet

		if R is None:
			# R = ArrayList(SimilarArray=U)
			R = np.copy(U)
		# Initialize residual to zero
		# R.SetUniformValue(0.)
		R[:] = 0.

		self.calculate_residual_bfaces(U, R)
		self.calculate_residual_elems(U, R)
		self.calculate_residual_ifaces(U, R)

		return R


	def apply_time_scheme(self, fhistory=None):
		'''
		Method: apply_time_scheme
		-----------------------------
		Applies the specified time scheme to update the solution

		'''

		EqnSet = self.EqnSet
		mesh = self.mesh
		Order = self.Params["InterpOrder"]
		Stepper = self.Stepper
		Time = self.Time

		# Parameters
		TrackOutput = self.Params["TrackOutput"]

		t0 = time.time()
		for iStep in range(self.nTimeStep):

			# Integrate in time
			# self.Time is used for local time
			R = Stepper.TakeTimeStep(self)

			# Increment time
			Time += Stepper.dt
			self.Time = Time

			# Info to print
			# PrintInfo = (iStep+1, self.Time, R.VectorNorm(ord=1))
			PrintInfo = (iStep+1, self.Time, np.linalg.norm(np.reshape(R,-1), ord=1))
			PrintString = "%d: Time = %g, Residual norm = %g" % (PrintInfo)

			# Output
			if TrackOutput:
				output,_ = Post.L2_error(mesh,EqnSet,Time,"Entropy",False)
				OutputString = ", Output = %g" % (output)
				PrintString += OutputString

			# Print info
			print(PrintString)

			# Write to file if requested
			if fhistory is not None:
				fhistory.write("%d %g %g" % (PrintInfo))
				if TrackOutput:
					fhistory.write(" %g" % (output))
				fhistory.write("\n")


		t1 = time.time()
		print("Wall clock time = %g seconds" % (t1-t0))


	def solve(self):
		'''
		Method: solve
		-----------------------------
		Performs the main solve of the DG method. Initializes the temporal loop. 

		'''
		mesh = self.mesh
		EqnSet = self.EqnSet

		OrderSequencing = self.Params["OrderSequencing"]
		InterpOrder = self.Params["InterpOrder"]
		nTimeStep = self.Params["nTimeStep"]
		EndTime = self.Params["EndTime"]
		WriteTimeHistory = self.Params["WriteTimeHistory"]
		if WriteTimeHistory:
			fhistory = open("TimeHistory.txt", "w")
		else:
			fhistory = None


		''' Convert to lists '''
		# InterpOrder
		if np.issubdtype(type(InterpOrder), np.integer):
			InterpOrders = [InterpOrder]
		elif type(InterpOrder) is list:
			InterpOrders = InterpOrder 
		else:
			raise TypeError
		nOrder = len(InterpOrders)
		# nTimeStep
		if np.issubdtype(type(nTimeStep), np.integer):
			nTimeSteps = [nTimeStep]*nOrder
		elif type(nTimeStep) is list:
			nTimeSteps = nTimeStep 
		else:
			raise TypeError
		# EndTime
		if np.issubdtype(type(EndTime), np.floating):
			EndTimes = []
			for i in range(nOrder):
				EndTimes.append(EndTime*(i+1))
		elif type(EndTime) is list:
			EndTimes = EndTime 
		else:
			raise TypeError


		''' Check compatibility '''
		if nOrder != len(nTimeSteps) or nOrder != len(EndTimes):
			raise ValueError

		if np.any(np.diff(EndTimes) < 0.):
			raise ValueError

		# if OrderSequencing:
		# 	if mesh.nElemGroup != 1:
		# 		# only compatible with 
		# 		raise Errors.IncompatibleError
		# else:
		# 	if len(InterpOrders) != 1:
		# 		raise ValueError

		if not OrderSequencing:
			if len(InterpOrders) != 1:
				raise ValueError



		''' Loop through Orders '''
		Time = 0.
		for iOrder in range(nOrder):
			Order = InterpOrders[iOrder]
			''' Compute time step '''
			self.Stepper.dt = (EndTimes[iOrder]-Time)/nTimeSteps[iOrder]
			self.nTimeStep = nTimeSteps[iOrder]

			''' After first iteration, project solution to next Order '''
			if iOrder > 0:
				# Clear DataSet
				delattr(self, "DataSet")
				self.DataSet = GenericData()
				# Increment Order
				Order_old = EqnSet.Order
				EqnSet.Order = Order
				# Project
				project_state_to_new_basis(self, mesh, EqnSet, EqnSet.Basis, Order_old)

			''' Apply time scheme '''
			self.apply_time_scheme(fhistory)

			Time = EndTimes[iOrder]


		if WriteTimeHistory:
			fhistory.close()

class ADERDG_Solver(DG_Solver):
	'''
	Class: ADERDG_Solver
	--------------------------------------------------------------------------
	Use the ADER-DG method to solve a given set of PDEs
	'''
	
	def check_solver_params(self):
		'''
		Method: check_solver_params
		--------------------------------------------------------------------------
		Checks the validity of the solver parameters

		'''
		Params = self.Params
		mesh = self.mesh
		EqnSet = self.EqnSet
		### Check interp basis validity
		if BasisType[Params["InterpBasis"]] == BasisType.LagrangeSeg or BasisType[Params["InterpBasis"]] == BasisType.LegendreSeg:
		    if mesh.Dim != 1:
		        raise Errors.IncompatibleError
		else:
		    if mesh.Dim != 2:
		        raise Errors.IncompatibleError

		### Check uniform mesh
		if Params["UniformMesh"] is True:
		    ''' 
		    Check that element volumes are the same
		    Note that this is a necessary but not sufficient requirement
		    '''
		    TotVol, ElemVols = MeshTools.element_volumes(mesh)
		    if (ElemVols.Max - ElemVols.Min)/TotVol > 1.e-8:
		        raise ValueError

		### Check linear geometric mapping
		if Params["LinearGeomMapping"] is True:
			if mesh.QOrder != 1:
			    raise Errors.IncompatibleError
			if mesh.QBasis == BasisType.LagrangeQuad \
			    and Params["UniformMesh"] is False:
			    raise Errors.IncompatibleError

		### Check limiter ###
		if Params["ApplyLimiter"] is 'ScalarPositivityPreserving' \
			and EqnSet.StateRank > 1:
				raise IncompatibleError
		if Params["ApplyLimiter"] is 'PositivityPreserving' \
			and EqnSet.StateRank == 1:
				raise IncompatibleError

		### Check time integration scheme ###
		TimeScheme = Params["TimeScheme"]
		if TimeScheme is not "ADER":
			raise Errors.IncompatibleError

		### Check flux/source coefficient interpolation compatability with basis functions.
		if Params["InterpolateFlux"] is True and BasisType[Params["InterpBasis"]] == BasisType.LegendreSeg:
			raise Errors.IncompatibleError

	def calculate_predictor_step(self, dt, W, Up):
		'''
		Method: calculate_predictor_step
		-------------------------------------------
		Calls the predictor step for each element

		INPUTS:
			dt: time step 
			W: previous time step solution in space only

		OUTPUTS:
			Up: predicted solution in space-time
		'''
		mesh = self.mesh
		EqnSet = self.EqnSet
		StaticData = None

		for elem in range(mesh.nElem):
			Up[elem], StaticData = self.calculate_predictor_elem(elem, dt, W[elem], Up[elem], StaticData)

		return Up

	def calculate_predictor_elem(self, elem, dt, W, Up, StaticData):
		'''
		Method: calculate_predictor_elem
		-------------------------------------------
		Calculates the predicted solution state for the ADER-DG method using a nonlinear solve of the
		weak form of the DG discretization in time.

		INPUTS:
			elem: element index
			dt: time step 
			W: previous time step solution in space only

		OUTPUTS:
			Up: predicted solution in space-time
		'''
	
		mesh = self.mesh
		EqnSet = self.EqnSet
		ns = EqnSet.StateRank
		dim = mesh.Dim
		basis = EqnSet.Basis #basis2
		basis_st = EqnSet.BasisADER #basis1
		order = EqnSet.Order
		entity = EntityType.Element

		quadData = None
		quadData_st = None
		QuadChanged = True; QuadChanged_st = True
		Elem2Nodes = mesh.Elem2Nodes[elem]
		dx = np.abs(mesh.Coords[Elem2Nodes[1],0]-mesh.Coords[Elem2Nodes[0],0])

		#Flux matrices in time
		FTL,_= get_temporal_flux_ader(mesh, basis_st, basis_st, order, elem=0, PhysicalSpace=False, StaticData=None)
		FTR,_= get_temporal_flux_ader(mesh, basis_st, basis, order, elem=0, PhysicalSpace=False, StaticData=None)
		
		#Stiffness matrix in time
		gradDir = 1
		SMT,_= get_stiffness_matrix_ader(mesh, basis_st, order, elem=0, gradDir=gradDir)
		gradDir = 0
		SMS,_= get_stiffness_matrix_ader(mesh, basis_st, order, elem=0, gradDir=gradDir)
		SMS = np.transpose(SMS)
		MM,_=  get_elem_mass_matrix_ader(mesh, basis_st, order, elem=-1, PhysicalSpace=False, StaticData=None)

		#MMinv,_= get_elem_inv_mass_matrix_ader(mesh, basis1, Order, PhysicalSpace=True, elem=-1, StaticData=None)

		QuadOrder,QuadChanged = get_gaussian_quadrature_elem(mesh, basis, order, EqnSet, quadData)
		QuadOrder_st, QuadChanged_st = get_gaussian_quadrature_elem(mesh, basis_st, order, EqnSet, quadData_st)
		QuadOrder_st-=1

		if QuadChanged:
			quadData = QuadData(mesh, basis, entity, QuadOrder)

		if QuadChanged_st:
			quadData_st = QuadData(mesh, basis_st, entity, QuadOrder_st)

		quad_pts = quadData.quad_pts
		quad_wts = quadData.quad_wts
		nq = quad_pts.shape[0]

		quad_pts_st = quadData_st.quad_pts
		quad_wts_st = quadData_st.quad_wts

		PhiData = BasisData(basis_st,order,mesh)
		PsiData = BasisData(basis,order,mesh)

		nb_st = order_to_num_basis_coeff(basis_st, order)
		nb   = order_to_num_basis_coeff(basis, order)
		'''
		BEGIN HACKY TEST (BRETT)
		# Hacky implementation of ADER-DG for linear solve only. Will delete long term.

		# MM = 0.*(dt/2.)*MM
		# SMS = 1.0*(dt/dx)*SMS
		# A1 = np.subtract(FTL,SMT)
		# A2 = np.add(A1,SMS)
		# ADER = np.add(A2,MM)

		# ADERinv = np.linalg.solve(ADER,FTR)


		# Up = np.matmul(ADERinv, W)

		# fluxpoly = self.flux_coefficients(elem, dt, Order, basis1, Up)
		# srcpoly = self.SourceCoefficients(elem, dt, Order, basis1, Up)

		END HACKY TEST (BRETT)
		'''

		#Make initial guess for the predictor step
		Up = Up.reshape(nb,nb,ns)

		for ir in range(ns):
			#for i in range(nn):
			for j in range(nb):
				Up[0,j,ir]=W[j,ir]
		#Up = np.pad(W,pad_width=((0,nnST-nn),(0,0)))
		Up = Up.reshape(nb_st,ns)

		def F(u):
			u = u.reshape(nb_st,ns)
			fluxpoly = self.flux_coefficients(elem, dt, order, basis_st, u)
			srcpoly = self.SourceCoefficients(elem, dt, order, basis_st, u)
			f = np.matmul(FTL,u)-np.matmul(FTR,W)-np.matmul(SMT,u)+np.matmul(SMS,fluxpoly)-np.matmul(MM,srcpoly)
			f = f.reshape(nb_st*ns)
			return f

		# def jac(u):
		# 	jac = np.zeros([nnST*sr,nnST*sr])
		# 	np.fill_diagonal(jac,1.)
		# 	return jac

		Up = Up.reshape(nb_st*ns)
		sol = root(F, Up, method='krylov') #non-linear solve for ADER-DG
		Up = sol.x
		Up = Up.reshape(nb_st,ns)

		return Up, StaticData

	def calculate_residual_elem(self, elem, U, ER, StaticData):
		'''
		Method: calculate_residual_elem
		-------------------------------------------
		Calculates the residual from the volume integral for each element

		INPUTS:
			elem: element index
			U: solution state

		OUTPUTS:
		ER: calculated residiual array (for volume integral of specified element)
		'''
		mesh = self.mesh
		EqnSet = self.EqnSet

		basis = EqnSet.Basis
		basis_st = EqnSet.BasisADER
		order = EqnSet.Order
		entity = EntityType.Element
		ns = EqnSet.StateRank
		dim = EqnSet.Dim

		if Order == 0:
			return ER, StaticData

		if StaticData is None:
			pnq = -1
			quadData = None
			quadData_st = None
			PhiData = None
			PsiData = None
			JData = JacobianData(mesh)
			xglob = None
			tglob = None
			u = None
			F = None
			s = None
			NData = None
			GeomPhiData = None
			TimePhiData = None
			StaticData = GenericData()
		else:
			nq = StaticData.pnq
			quadData = StaticData.quadData
			quadData_st = StaticData.quadData_st
			PhiData = StaticData.PhiData
			PsiData = StaticData.PsiData
			JData = StaticData.JData
			xglob = StaticData.xglob
			tglob = StaticData.tglob
			u = StaticData.u
			F = StaticData.F
			s = StaticData.s
			NData = StaticData.NData
			GeomPhiData = StaticData.GeomPhiData
			TimePhiData = StaticData.TimePhiData

		QuadOrder,QuadChanged = get_gaussian_quadrature_elem(mesh, mesh.QBasis, order, EqnSet, quadData)
		QuadOrder_st, QuadChanged_st = get_gaussian_quadrature_elem(mesh, basis_st, order, EqnSet, quadData_st)
		QuadOrder_st-=1

		if QuadChanged:
			quadData = QuadData(mesh, basis, entity, QuadOrder)
		if QuadChanged_st:
			quadData_st = QuadData(mesh, basis_st, entity, QuadOrder_st)

		quad_pts = quadData.quad_pts
		quad_wts = quadData.quad_wts
		nq = quad_pts.shape[0]

		quad_pts_st = quadData_st.quad_pts
		quad_wts_st = quadData_st.quad_wts
		nq_st = quad_pts_st.shape[0]

		if QuadChanged:
			PhiData = BasisData(basis_st,order,mesh)
			PhiData.eval_basis(quad_pts_st, Get_Phi=True, Get_GPhi=False)
			PsiData = BasisData(basis,order,mesh)
			PsiData.eval_basis(quad_pts, Get_Phi=True, Get_GPhi=True)

			xglob = np.zeros([nq, dim])
			u = np.zeros([nq_st, ns])
			F = np.zeros([nq_st, ns, dim])
			s = np.zeros([nq_st, ns])

		JData.element_jacobian(mesh,elem,quad_pts,get_djac=True,get_jac=False,get_ijac=True)
		PsiData.eval_basis(quad_pts, Get_gPhi=True, JData=JData)

		xglob, GeomPhiData = ref_to_phys(mesh, elem, GeomPhiData, quad_pts, xglob, QuadChanged)
		tglob, TimePhiData = ref_to_phys_time(mesh, elem, self.Time, self.Stepper.dt, TimePhiData, quad_pts_st, tglob, QuadChanged_st)


		#nn = PsiData.nn
		nb = PsiData.Phi.shape[1]
		# interpolate state and gradient at quad points
		u[:] = np.matmul(PhiData.Phi, U)

		F = EqnSet.ConvFluxInterior(u, F) # [nq,sr,dim]

		# F = np.reshape(F,(nq,nq,sr,dim))

		# for ir in range(sr):
		# 	for k in range(nn): # Loop over basis function in space
		# 		for i in range(nq): # Loop over time
		# 			for j in range(nq): # Loop over space
		# 				gPsi = PsiData.gPhi[j,k]
		# 				ER[k,ir] += wq[i]*wq[j]*JData.djac[j*(JData.nq!=1)]*F[i,j,ir]*gPsi

		# F = np.reshape(F,(nqST,sr,dim))
		
		ER[:] += np.tensordot(np.tile(PsiData.gPhi,(nb,1,1)), F*(quad_wts_st.reshape(nq,nq)*JData.djac).reshape(nq_st,1,1), axes=([0,2],[0,2])) # [nn, sr]


		s = np.zeros([nq_st,ns])
		s = EqnSet.SourceState(nq_st, xglob, tglob, NData, u, s) # [nq,sr,dim]

		# s = np.reshape(s,(nq,nq,sr))
		# #Calculate source term integral
		# for ir in range(sr):
		# 	for k in range(nn):
		# 		for i in range(nq): # Loop over time
		# 			for j in range(nq): # Loop over space
		# 				Psi = PsiData.Phi[j,k]
		# 				ER[k,ir] += wq[i]*wq[j]*s[i,j,ir]*JData.djac[j*(JData.nq!=1)]*Psi
		# s = np.reshape(s,(nqST,sr))

		ER[:] += np.matmul(np.tile(PsiData.Phi,(nb,1)).transpose(),s*(quad_wts_st.reshape(nq,nq)*JData.djac).reshape(nq_st,1))

		if elem == echeck:
			code.interact(local=locals())

		# Store in StaticData
		StaticData.pnq = nq
		StaticData.quadData = quadData
		StaticData.quadData_st = quadData_st
		StaticData.PhiData = PhiData
		StaticData.PsiData = PsiData
		StaticData.JData = JData
		StaticData.xglob = xglob
		StaticData.tglob = tglob
		StaticData.u = u
		StaticData.F = F
		StaticData.s = s
		StaticData.NData = NData
		StaticData.GeomPhiData = GeomPhiData
		StaticData.TimePhiData = TimePhiData

		return ER, StaticData

	def calculate_residual_iface(self, iiface, UL, UR, RL, RR, StaticData):
		'''
		Method: calculate_residual_iface
		-------------------------------------------
		Calculates the residual from the boundary integral for each internal face

		INPUTS:
			iiface: internal face index
			UL: solution array from left neighboring element
			UR: solution array from right neighboring element
			
		OUTPUTS:
			RL: calculated residual array (left neighboring element contribution)
			RR: calculated residual array (right neighboring element contribution)
		'''
		mesh = self.mesh
		dim = mesh.Dim
		EqnSet = self.EqnSet
		ns = EqnSet.StateRank

		IFace = mesh.IFaces[iiface]
		elemL = IFace.ElemL
		elemR = IFace.ElemR
		faceL = IFace.faceL
		faceR = IFace.faceR
		order = EqnSet.Order
		nFacePerElem = mesh.nFacePerElem
		nFacePerElemADER = nFacePerElem + 2 # Hard-coded for the 1D ADER method

		if StaticData is None:
			pnq = -1
			quadData = None
			quadData_st = None
			PhiDataL = None
			PhiDataR = None
			PsiDataL = None
			PsiDataR = None
			xelemLPhi = None
			xelemRPhi = None
			xelemLPsi = None
			xelemRPsi = None
			uL = None
			uR = None
			StaticData = GenericData()
			NData = None
		else:
			nq = StaticData.pnq
			quadData_st = StaticData.quadData_st
			quadData = StaticData.quadData
			Faces2PhiDataL = StaticData.Faces2PhiDataL
			Faces2PhiDataR = StaticData.Faces2PhiDataR
			Faces2PsiDataL = StaticData.Faces2PsiDataL
			Faces2PsiDataR = StaticData.Faces2PsiDataR
			uL = StaticData.uL
			uR = StaticData.uR
			NData = StaticData.NData


		basis = EqnSet.Basis
		basis_st = EqnSet.BasisADER

		QuadOrder_st, QuadChanged_st = get_gaussian_quadrature_face(mesh, IFace, basis_st, order, EqnSet, quadData_st)
		QuadOrder_st-=1
		QuadOrder, QuadChanged = get_gaussian_quadrature_face(mesh, IFace, mesh.QBasis, order, EqnSet, quadData)

		if QuadChanged:
			quadData = QuadData(mesh, EqnSet.Basis, EntityType.IFace, QuadOrder)
		if QuadChanged_st:
			quadData_st = QuadDataADER(mesh,basis_st,EntityType.IFace, QuadOrder_st)

		quad_pts = quadData.quad_pts
		quad_wts = quadData.quad_wts
		nq = quad_pts.shape[0]

		quad_pts_st = quadData_st.quad_pts
		quad_wts_st = quadData_st.quad_wts
		nq_st = quad_pts_st.shape[0]

		if faceL == 0:
			face_stL = 3
		elif faceL == 1:
			face_stL = 1
		else:
			return IncompatibleError
		if faceR == 0:
			face_stR = 3
		elif faceR == 1:
			face_stR = 1
		else:
			return IncompatibleError

		if QuadChanged or QuadChanged_st:

			Faces2PsiDataL = [None for i in range(nFacePerElem)]
			Faces2PsiDataR = [None for i in range(nFacePerElem)]
			Faces2PhiDataL = [None for i in range(nFacePerElemADER)]
			Faces2PhiDataR = [None for i in range(nFacePerElemADER)]
			PsiDataL = BasisData(basis, order, mesh)
			PsiDataR = BasisData(basis, order, mesh)
			PhiDataL = BasisData(basis_st, order, mesh)
			PhiDataR = BasisData(basis_st, order, mesh)

			xglob = np.zeros([nq, ns])
			uL = np.zeros([nq_st, ns])
			uR = np.zeros([nq_st, ns])

			xelemLPhi = np.zeros([nq_st, dim+1])
			xelemRPhi = np.zeros([nq_st, dim+1])
			xelemLPsi = np.zeros([nq, dim])
			xelemRPsi = np.zeros([nq, dim])
			
		PsiDataL = Faces2PhiDataL[faceL]
		PsiDataR = Faces2PsiDataR[faceR]
		PhiDataL = Faces2PsiDataL[faceL]
		PhiDataR = Faces2PhiDataR[faceR]

		#Evaluate space-time basis functions
		if PhiDataL is None or QuadChanged_st:
			Faces2PhiDataL[faceL] = PhiDataL = BasisData(basis_st,order,mesh)
			xelemLPhi = PhiDataL.eval_basis_on_face_ader(mesh, basis_st, face_stL, quad_pts_st, xelemLPhi, Get_Phi=True)
		if PhiDataR is None or QuadChanged_st:
			Faces2PhiDataR[faceR] = PhiDataR = BasisData(basis_st,order,mesh)
			xelemRPhi = PhiDataR.eval_basis_on_face_ader(mesh, basis_st, face_stR, quad_pts_st[::-1], xelemRPhi, Get_Phi=True)

		#Evaluate spacial basis functions
		if PsiDataL is None or QuadChanged:
			Faces2PsiDataL[faceL] = PsiDataL = BasisData(basis,order,mesh)
			xelemLPsi = PsiDataL.eval_basis_on_face(mesh, faceL, quad_pts, xelemLPsi, Get_Phi=True)
		if PsiDataR is None or QuadChanged:
			Faces2PsiDataR[faceR] = PsiDataR = BasisData(basis,order,mesh)
			xelemRPsi = PsiDataR.eval_basis_on_face(mesh, faceR, quad_pts[::-1], xelemRPsi, Get_Phi=True)

		NData = iface_normal(mesh, IFace, quad_pts, NData)

		nbL = PsiDataL.Phi.shape[1]
		nbR = PsiDataR.Phi.shape[1]
		nb = np.amax([nbL,nbR])

		Stepper = self.Stepper
		dt = Stepper.dt

		uL[:] = np.matmul(PhiDataL.Phi, UL)
		uR[:] = np.matmul(PhiDataR.Phi, UR)

		F = EqnSet.ConvFluxNumerical(uL, uR, NData, nq_st, StaticData) # [nq,sr]

		# F = np.reshape(F,(nq,nqST,sr))
		
		# for ir in range(sr):
		# 	#for k in range(nn): # Loop over basis function in space
		# 	for i in range(nqST): # Loop over time
		# 		for j in range(nq): # Loop over space
		# 			PsiL = PsiDataL.Phi[j,:]
		# 			PsiR = PsiDataR.Phi[j,:]
		# 			RL[:,ir] -= wqST[i]*wq[j]*F[j,i,ir]*PsiL
		# 			RR[:,ir] += wqST[i]*wq[j]*F[j,i,ir]*PsiR

		# F = np.reshape(F,(nqST,sr))


		RL -= np.matmul(np.tile(PsiDataL.Phi,(nb,1)).transpose(), F*quad_wts_st*quad_wts)
		RR += np.matmul(np.tile(PsiDataR.Phi,(nb,1)).transpose(), F*quad_wts_st*quad_wts)

		if elemL == echeck or elemR == echeck:
			if elemL == echeck: print("Left!")
			else: print("Right!")
			code.interact(local=locals())

		# Store in StaticData
		StaticData.pnq = nq
		StaticData.quadData = quadData
		StaticData.quadData_st = quadData_st
		StaticData.Faces2PhiDataL = Faces2PhiDataL
		StaticData.Faces2PhiDataR = Faces2PhiDataR
		StaticData.Faces2PsiDataL = Faces2PsiDataL
		StaticData.Faces2PsiDataR = Faces2PsiDataR
		StaticData.uL = uL
		StaticData.uR = uR
		StaticData.NData = NData

		return RL, RR, StaticData

	def calculate_residual_bface(self, ibfgrp, ibface, U, R, StaticData):
		'''
		Method: calculate_residual_bface
		-------------------------------------------
		Calculates the residual from the boundary integral for each boundary face

		INPUTS:
			ibfgrp: index of BC group
			ibface: index of boundary face
			U: solution array from internal element
			
		OUTPUTS:
			R: calculated residual array (from boundary face)
		'''
		mesh = self.mesh
		EqnSet = self.EqnSet
		ns = EqnSet.StateRank
		dim = mesh.Dim

		BFG = mesh.BFaceGroups[ibfgrp]
		BFace = BFG.BFaces[ibface]
		elem = BFace.Elem
		face = BFace.face
		order = EqnSet.Order
		nFacePerElem = mesh.nFacePerElem
		nFacePerElemADER = nFacePerElem + 2 # Hard-coded for 1D ADER method

		if StaticData is None:
			pnq = -1
			quadData = None
			quadData_st = None
			PhiData = None
			PsiData = None
			xglob = None
			tglob = None
			uI = None
			uB = None
			NData = None
			GeomPhiData = None
			TimePhiData = None
			xelemPsi = None
			xelemPhi = None
			StaticData = GenericData()

		else:
			nq = StaticData.pnq
			quadData = StaticData.quadData
			quadData_st = StaticData.quadData_st
			PhiData = StaticData.PhiData
			PsiData = StaticData.PsiData
			xglob = StaticData.xglob
			tglob = StaticData.tglob
			uI = StaticData.uI
			uB = StaticData.uB
			NData = StaticData.NData
			GeomPhiData = StaticData.GeomPhiData
			TimePhiData = StaticData.TimePhiData
			Faces2PhiData = StaticData.Faces2PhiData
			Faces2PsiData = StaticData.Faces2PsiData
			Faces2xelemADER = StaticData.Faces2xelemADER
			Faces2xelem = StaticData.Faces2xelem
		
		basis = EqnSet.Basis
		basis_st = EqnSet.BasisADER

		QuadOrder_st, QuadChanged_st = get_gaussian_quadrature_face(mesh, BFace, basis_st, order, EqnSet, quadData_st)
		QuadOrder_st-=1
		QuadOrder, QuadChanged = get_gaussian_quadrature_face(mesh, BFace, mesh.QBasis, order, EqnSet, quadData)

		if QuadChanged:
			quadData = QuadData(mesh, EqnSet.Basis, EntityType.BFace, QuadOrder)
		if QuadChanged_st:
			quadData_st = QuadDataADER(mesh,basis_st,EntityType.BFace, QuadOrder_st)
		
		quad_pts = quadData.quad_pts
		quad_wts = quadData.quad_wts
		nq = quad_pts.shape[0]

		quad_pts_st = quadData_st.quad_pts
		quad_wts_st = quadData_st.quad_wts
		nq_st = quad_pts_st.shape[0]

		if face == 0:
			face_st = 3
		elif face == 1:
			face_st = 1
		else:
			return IncompatibleError

		if QuadChanged or QuadChanged_st:
			Faces2PsiData = [None for i in range(nFacePerElem)]
			Faces2PhiData = [None for i in range(nFacePerElemADER)]
			PsiData = BasisData(basis, order, mesh)
			PhiData = BasisData(basis_st, order, mesh)
			xglob = np.zeros([nq, ns])
			uI = np.zeros([nq_st, ns])
			uB = np.zeros([nq_st, ns])
			Faces2xelem = np.zeros([nFacePerElem, nq, dim])
			Faces2xelemADER = np.zeros([nFacePerElem, nq_st, dim+1])

		PsiData = Faces2PhiData[face]
		PhiData = Faces2PsiData[face]
		xelemPsi = Faces2xelem[face]
		xelemPhi = Faces2xelemADER[face]

		if PsiData is None or QuadChanged:
			Faces2PsiData[face] = PsiData = BasisData(basis,order,mesh)
			Faces2xelem[face] = xelemPsi = PsiData.eval_basis_on_face(mesh, face, quad_pts, xelemPsi, Get_Phi=True)
		if PhiData is None or QuadChanged_st:
			Faces2PhiData[face] = PhiData = BasisData(basis_st,order,mesh)
			Faces2xelemADER[face] = xelemPhi = PhiData.eval_basis_on_face_ader(mesh, basis_st, face_st, quad_pts_st, xelemPhi, Get_Phi=True)


		NData = bface_normal(mesh, BFace, quad_pts, NData)
		PointsChanged = QuadChanged or face != GeomPhiData.face
		xglob, GeomPhiData = ref_to_phys(mesh, elem, GeomPhiData, xelemPsi, xglob, PointsChanged)

		tglob, TimePhiData = ref_to_phys_time(mesh, elem, self.Time, self.Stepper.dt, TimePhiData, xelemPhi, tglob, PointsChanged)
		#nn = PsiData.nn
		nb = PsiData.Phi.shape[1]

		# interpolate state and gradient at quad points
		uI[:] = np.matmul(PhiData.Phi, U)

		# Get boundary state
		BC = EqnSet.BCs[ibfgrp]
		uB = EqnSet.BoundaryState(BC, nq_st, xglob, tglob, NData, uI, uB)

		F = EqnSet.ConvFluxBoundary(BC, uI, uB, NData, nq_st, StaticData) # [nq,sr]

		# F = np.reshape(F,(nq,nqST,sr))

		# for ir in range(sr):
		# 	for i in range(nqST): # Loop over time
		# 		for j in range(nq): # Loop over space
		# 			Psi = PsiData.Phi[j,:]
		# 			R[:,ir] -= wqST[i]*wq[j]*F[j,i,ir]*Psi
	
		# F = np.reshape(F,(nqST,sr))

		R[:] -= np.matmul(np.tile(PsiData.Phi,(nb,1)).transpose(),F*quad_wts_st*quad_wts)

		if elem == echeck:
			code.interact(local=locals())

		# Store in StaticData
		StaticData.pnq = nq
		StaticData.quadData = quadData
		StaticData.quadData_st = quadData_st
		StaticData.PhiData = PhiData
		StaticData.PsiData = PsiData
		StaticData.uI = uI
		StaticData.uB = uB
		StaticData.F = F
		StaticData.xglob = xglob
		StaticData.tglob = tglob
		StaticData.NData = NData
		StaticData.xelemPsi = xelemPsi
		StaticData.xelemPhi = xelemPhi
		StaticData.GeomPhiData = GeomPhiData
		StaticData.TimePhiData = TimePhiData
		StaticData.Faces2PhiData = Faces2PhiData
		StaticData.Faces2PsiData = Faces2PsiData
		StaticData.Faces2xelemADER = Faces2xelemADER
		StaticData.Faces2xelem = Faces2xelem

		return R, StaticData


	def flux_coefficients(self, elem, dt, Order, basis, U):
		'''
		Method: flux_coefficients
		-------------------------------------------
		Calculates the polynomial coefficients for the flux functions in ADER-DG

		INPUTS:
			elem: element index
			dt: time step size
			order: 
			U: solution array
			
		OUTPUTS:
			F: polynomical coefficients of the flux function
		'''
		mesh = self.mesh
		dim = mesh.Dim
		EqnSet = self.EqnSet
		sr = EqnSet.StateRank
		Params = self.Params
		entity = EntityType.Element
		InterpolateFlux = Params["InterpolateFlux"]

		quadData = None
		quadDataST = None
		JData = JacobianData(mesh)
		GeomPhiData = None
		xq = None; xphys = None; QuadChanged = True; QuadChangedST = True;

		rhs = np.zeros([order_to_num_basis_coeff(basis,Order),sr],dtype=U.dtype)

		if not InterpolateFlux:


			QuadOrder,QuadChanged = get_gaussian_quadrature_elem(mesh, mesh.QBasis, Order, EqnSet, quadData)
			QuadOrderST, QuadChangedST = get_gaussian_quadrature_elem(mesh, basis, Order, EqnSet, quadDataST)
			QuadOrderST-=1

			if QuadChanged:
				quadData = QuadData(mesh, mesh.QBasis, entity, QuadOrder)
			if QuadChangedST:
				quadDataST = QuadData(mesh, basis, entity, QuadOrderST)

			nq = quadData.nquad
			xq = quadData.quad_pts
			wq = quadData.quad_wts

			nqST = quadDataST.nquad
			xqST = quadDataST.quad_pts
			wqST = quadDataST.quad_wts

			if QuadChanged:

				PhiData = BasisData(basis,Order,mesh)
				PhiData.eval_basis(xqST, Get_Phi=True)
				xphys = np.zeros([nqST, mesh.Dim])

			JData.element_jacobian(mesh,elem,xqST,get_djac=True)
			MMinv,_= get_elem_inv_mass_matrix_ader(mesh, basis, Order, elem=-1, PhysicalSpace=True, StaticData=None)
			nn = PhiData.nn
		else:

			xq, nq = equidistant_nodes(basis, Order, xq)
			nn = nq
			JData.element_jacobian(mesh,elem,xq,get_djac=True)

		if not InterpolateFlux:

			u = np.zeros([nqST, sr])
			u[:] = np.matmul(PhiData.Phi, U)

			#u = np.reshape(u,(nq,nn))
			F = np.zeros([nqST,sr,dim])
			f = EqnSet.ConvFluxInterior(u, F) # [nq,sr]
			f = f.reshape(nqST,sr)
			#f = np.reshape(f,(nq,nq,sr,dim))
			#Phi = PhiData.Phi
			#Phi = np.reshape(Phi,(nq,nq,nn))
		
			rhs *=0.
			# for ir in range(sr):
			# 	for k in range(nn): # Loop over basis function in space
			# 		for i in range(nq): # Loop over time
			# 			for j in range(nq): # Loop over space
			# 				#Phi = PhiData.Phi[j,k]
			# 				rhs[k,ir] += wq[i]*wq[j]*JData.djac[j*(JData.nq!=1)]*f[i,j,ir]*Phi[i,j,k]

			rhs[:] = np.matmul(PhiData.Phi.transpose(), f*(wqST*JData.djac))


			#F = np.reshape(F,(nqST,sr,dim))
			F = np.dot(MMinv,rhs)*(1.0/JData.djac)*dt/2.0

		else:
			F1 = np.zeros([nn,sr,dim])
			#code.interact(local=locals())
			f = EqnSet.ConvFluxInterior(U,F1)
			F = f[:,:,0]*(1.0/JData.djac)*dt/2.0
			#code.interact(local=locals())
		return F

	def SourceCoefficients(self, elem, dt, Order, basis, U):

		mesh = self.mesh
		dim = mesh.Dim
		EqnSet = self.EqnSet
		sr = EqnSet.StateRank
		Params = self.Params
		entity = EntityType.Element
		InterpolateFlux = Params["InterpolateFlux"]

		quadData = None
		quadDataST = None
		JData = JacobianData(mesh)
		GeomPhiData = None
		TimePhiData = None
		xq = None; xphys = None; QuadChanged = True; QuadChangedST = True;
		xglob = None; tglob = None; NData = None;

		rhs = np.zeros([order_to_num_basis_coeff(basis,Order),sr],dtype=U.dtype)

		if not InterpolateFlux:

			QuadOrder,QuadChanged = get_gaussian_quadrature_elem(mesh, mesh.QBasis, Order, EqnSet, quadData)
			QuadOrderST, QuadChangedST = get_gaussian_quadrature_elem(mesh, basis, Order, EqnSet, quadDataST)
			QuadOrderST-=1

			if QuadChanged:
				quadData = QuadData(mesh, mesh.QBasis, entity, QuadOrder)
			if QuadChangedST:
				quadDataST = QuadData(mesh, basis, entity, QuadOrderST)

			nq = quadData.nquad
			xq = quadData.quad_pts
			wq = quadData.quad_wts

			nqST = quadDataST.nquad
			xqST = quadDataST.quad_pts
			wqST = quadDataST.quad_wts

			xglob, GeomPhiData = ref_to_phys(mesh, elem, GeomPhiData, xq, xglob, QuadChanged)
			tglob, TimePhiData = ref_to_phys_time(mesh, elem, self.Time, self.Stepper.dt, TimePhiData, xqST, tglob, QuadChangedST)
			
			if QuadChanged:

				PhiData = BasisData(basis,Order,mesh)
				PhiData.eval_basis(xqST, Get_Phi=True)
				xphys = np.zeros([nqST, mesh.Dim])

			JData.element_jacobian(mesh,elem,xqST,get_djac=True)
			MMinv,_= get_elem_inv_mass_matrix_ader(mesh, basis, Order, elem=-1, PhysicalSpace=True, StaticData=None)
			nn = PhiData.nn
		else:

			xq, nq = equidistant_nodes(mesh.QBasis, Order, xq)
			nn = nq
			JData.element_jacobian(mesh,elem,xq,get_djac=True)
			
			QuadOrderST, QuadChangedST = get_gaussian_quadrature_elem(mesh, basis, Order, EqnSet, quadDataST)
			QuadOrderST-=1

			if QuadChangedST:
				quadDataST = QuadData(mesh, basis, entity, QuadOrderST)

			nqST = quadDataST.nquad
			xqST = quadDataST.quad_pts

			xglob, GeomPhiData = ref_to_phys(mesh, elem, GeomPhiData, xq, xglob, QuadChanged)
			tglob, TimePhiData = ref_to_phys_time(mesh, elem, self.Time, self.Stepper.dt, TimePhiData, xqST, tglob, QuadChangedST)

		if not InterpolateFlux:

			u = np.zeros([nqST, sr])
			u[:] = np.matmul(PhiData.Phi, U)

			#u = np.reshape(u,(nq,nn))
			#S = np.zeros([nqST,sr,dim])
			s = np.zeros([nqST,sr])
			s = EqnSet.SourceState(nqST, xglob, tglob, NData, u, s) # [nq,sr]
			
			#s = np.reshape(s,(nq,nq,sr,dim))
			#Phi = PhiData.Phi
			#Phi = np.reshape(Phi,(nq,nq,nn))
		
			rhs *=0.
			# for ir in range(sr):
			# 	for k in range(nn): # Loop over basis function in space
			# 		for i in range(nq): # Loop over time
			# 			for j in range(nq): # Loop over space
			# 				#Phi = PhiData.Phi[j,k]
			# 				rhs[k,ir] += wq[i]*wq[j]*JData.djac[j*(JData.nq!=1)]*s[i,j,ir]*Phi[i,j,k]
			rhs[:] = np.matmul(PhiData.Phi.transpose(),s*wqST*JData.djac)
			S = np.dot(MMinv,rhs)*dt/2.0

		else:
			S1 = np.zeros([nqST,sr])
			s = EqnSet.SourceState(nqST, xglob, tglob, NData, U, S1)
			S = s*dt/2.0
		return S
