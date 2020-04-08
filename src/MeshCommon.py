from General import BasisType
from Mesh import *


def mesh_1D(Coords=None, nElem=10, Uniform=True, xmin=-1., xmax=1., Periodic=True):
	'''
	Function: mesh_1D
	-------------------
	This function creates a 1D mesh.

	INPUTS:
	    Coords: x-coordinates
	    Uniform: True for a uniform mesh (will be set to False if Coords is not None)
	    nElem: number of elements (only relevant for Uniform=True)
	    xmin: minimum coordinate (only relevant for Uniform=True)
	    xmax: maximum coordinate (only relevant for Uniform=True)
	    Periodic: True for a periodic mesh

	OUTPUTS:
	    mesh: Mesh object that stores relevant mesh info
	'''
	if Coords is None and not Uniform:
		raise Exception("Input error")

	### Create mesh
	if Coords is None:
		nNode = nElem + 1
		mesh = Mesh(dim=1, nNode=nNode)
		mesh.Coords = np.zeros([mesh.nNode,mesh.Dim])
		mesh.Coords[:,0] = np.linspace(xmin,xmax,mesh.nNode)
	else:
		Uniform = False
		Coords.shape = -1,1
		nNode = Coords.shape[0]
		nElem = nNode - 1
		mesh = Mesh(dim=1, nNode=nNode)
		mesh.Coords = Coords

	# IFaces
	if Periodic:
		mesh.nIFace = mesh.nNode - 1
		mesh.allocate_ifaces()
		# mesh.IFaces = [IFace() for i in range(mesh.nIFace)]
		for i in range(mesh.nIFace):
			IFace_ = mesh.IFaces[i]
			IFace_.ElemL = i-1
			IFace_.faceL = 1
			IFace_.ElemR = i
			IFace_.faceR = 0
		# Leftmost face
		mesh.IFaces[0].ElemL = nElem - 1
	# Rightmost face
	# mesh.IFaces[-1].ElemR = 0
	else:
		mesh.nIFace = nElem - 1
		mesh.allocate_ifaces()
		for i in range(mesh.nIFace):
			IFace_ = mesh.IFaces[i]
			IFace_.ElemL = i
			IFace_.faceL = 1
			IFace_.ElemR = i+1
			IFace_.faceR = 0
		# Boundary groups
		mesh.nBFaceGroup = 2
		mesh.allocate_bface_groups()
		for i in range(mesh.nBFaceGroup):
			BFG = mesh.BFaceGroups[i]
			BFG.nBFace = 1
			BFG.allocate_bfaces()
			BF = BFG.BFaces[0]
			if i == 0:
				BFG.Name = "Left"
				BF.Elem = 0
				BF.face = 0
			else:
				BFG.Name = "Right"
				BF.Elem = nElem - 1
				BF.face = 1

	mesh.SetParams(QBasis=BasisType["LagrangeSeg"], QOrder=1, nElem=nElem)
	mesh.allocate_faces()
	# interior elements
	for elem in range(mesh.nElem):
		for i in range(mesh.nFacePerElem):
			Face_ = mesh.Faces[elem][i]
			Face_.Type = INTERIORFACE
			Face_.Number = elem + i
			if not Periodic:
				if elem == 0 and i == 0:
					Face_.Type = NULLFACE
					Face_.Number = 0
				elif elem == mesh.nElem-1 and i == 1:
					Face_.Type = NULLFACE
					Face_.Number = 1
				else:
					Face_.Number = elem + i - 1


	mesh.allocate_elem_to_nodes()
	for elem in range(mesh.nElem):
		for i in range(mesh.nNodePerElem):
			mesh.Elem2Nodes[elem][i] = elem + i

	mesh.allocate_helpers()
	mesh.fill_faces()

	return mesh


def refine_uniform_1D(Coords_old):
	'''
	Function: refine_uniform_1D
	-------------------
	This function uniformly refines a set of coordinates

	INPUTS:
	    Coords_old: coordinates to refine

	OUTPUTS:
	    Coords: refined coordinates
	'''
	nNode_old = len(Coords_old)
	nElem_old = nNode_old-1

	nElem = nElem_old*2
	nNode = nElem+1

	Coords = np.zeros([nNode,1])

	for n in range(nNode_old-1):
		Coords[2*n] = Coords_old[n]
		Coords[2*n+1] = np.mean(Coords_old[n:n+2])
	Coords[-1] = Coords_old[-1]

	return Coords










