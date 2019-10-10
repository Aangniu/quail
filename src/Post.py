import numpy as np
import copy
import code
from Quadrature import GetQuadOrderElem, QuadData
from Basis import BasisData, JacobianData
from Mesh import Mesh, Ref2Phys
from General import *
import MeshTools
from Data import ArrayList


def L2_error(mesh,EqnSet,EndTime,VariableName):

	U = EqnSet.U.Arrays

	# Check for exact solution
	if not EqnSet.ExactSoln.Function:
		raise Exception("No exact solution provided")

	# Get elem volumes 
	TotVol,_ = MeshTools.ElementVolumes(mesh)

	# Get error
	# ElemErr = copy.deepcopy(U)
	# ElemErr = ArrayList(SimilarArray=EqnSet.U).Arrays
	ElemErr = ArrayList(nArray=mesh.nElemGroup,ArrayDims=[mesh.nElems])
	TotErr = 0.
	sr = EqnSet.StateRank
	quadData = None
	JData = JacobianData(mesh)
	ier = EqnSet.VariableType[VariableName]
	for egrp in range(mesh.nElemGroup):
		ElemErr.Arrays[egrp][:] = 0.

		Order = EqnSet.Orders[egrp]
		basis = EqnSet.Bases[egrp]

		for elem in range(mesh.nElems[egrp]):
			U_ = U[egrp][elem]

			QuadOrder,QuadChanged = GetQuadOrderElem(mesh, egrp, basis, 2*np.amax([Order,1]), EqnSet, quadData)
			if QuadChanged:
				quadData = QuadData(mesh, egrp, EntityType.Element, QuadOrder)

			nq = quadData.nquad
			xq = quadData.xquad
			wq = quadData.wquad

			if QuadChanged:
				# PhiData = BasisData(egrp,Order,entity,nq,xq,mesh,True,True)
				PhiData = BasisData(basis,Order,nq,mesh)
				PhiData.EvalBasis(xq, True, False, False, None)
				xphys = np.zeros([nq, mesh.Dim])

			JData.ElemJacobian(egrp,elem,nq,xq,mesh,Get_detJ=True)

			xphys = Ref2Phys(mesh, egrp, elem, PhiData, nq, xq, xphys)
			u_exact = EqnSet.CallFunction(EqnSet.ExactSoln, x=xphys, Time=EndTime)

			# interpolate state at quad points
			u = np.zeros([nq, sr])
			for ir in range(sr):
				u[:,ir] = np.matmul(PhiData.Phi, U_[:,ir])

			err = 0.
			for iq in range(nq):
				err += (u[iq,ier] - u_exact[iq,ier])**2.*wq[iq] * JData.detJ[iq*(JData.nq != 1)]
			ElemErr.Arrays[egrp][elem] = err
			TotErr += ElemErr.Arrays[egrp][elem]

	# TotErr /= TotVol
	TotErr = np.sqrt(TotErr/TotVol)

	# print("Total volume = %g" % (TotVol))
	print("Total error = %g" % (TotErr))

	return TotErr, ElemErr