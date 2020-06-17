import code
import copy

import general

from meshing.meshbase import *
import numerics.basis.basis as Basis

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

	mesh.SetParams(gbasis=Basis.LagrangeEqSeg(1), gorder=1, nElem=nElem)
	mesh.allocate_faces()
	# interior elements
	for elem in range(mesh.nElem):
		for i in range(mesh.nFacePerElem):
			Face_ = mesh.Faces[elem][i]
			Face_.Type = general.INTERIORFACE
			Face_.Number = elem + i
			if not Periodic:
				if elem == 0 and i == 0:
					Face_.Type = general.NULLFACE
					Face_.Number = 0
				elif elem == mesh.nElem-1 and i == 1:
					Face_.Type = general.NULLFACE
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


def mesh_2D(xcoords=None, ycoords=None, nElem_x=10, nElem_y = 10, Uniform=True, xmin=-1., xmax=1., 
	ymin=-1., ymax=1., Periodic=True):
	'''
	Function: mesh_2D
	-------------------
	This function creates a 2D mesh.

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

	### Create mesh
	if xcoords is None and ycoords is None:
		# Uniform
		nNode_x = nElem_x + 1
		nNode_y = nElem_y + 1
		xcoords = np.linspace(xmin, xmax, nNode_x)
		ycoords = np.linspace(ymin, ymax, nNode_y)
	elif xcoords is not None and ycoords is not None:
		Uniform = False
		nNode_x = len(xcoords)
		nNode_y = len(ycoords)
		nElem_x = nNode_x - 1
		nElem_y = nNode_y - 1
	else:
		raise Exception("Input error")

	X, Y = np.meshgrid(xcoords, ycoords)
	xp = np.array([np.reshape(X,-1),np.reshape(Y,-1)]).transpose()

	mesh = Mesh(dim=2, nNode=xp.shape[0], nElem=nElem_x*nElem_y, gbasis=Basis.LagrangeEqQuad(1),
		gorder=1)

	mesh.Coords = xp

	### Elems
	mesh.allocate_elem_to_nodes()
	elem = 0
	for ny in range(nElem_y):
		for nx in range(nElem_x):
			mesh.Elem2Nodes[elem][0] = nNode_x*ny + nx
			mesh.Elem2Nodes[elem][1] = nNode_x*ny + nx + 1
			mesh.Elem2Nodes[elem][2] = nNode_x*(ny+1) + nx
			mesh.Elem2Nodes[elem][3] = nNode_x*(ny+1) + nx + 1
			elem += 1

	mesh.allocate_faces()

	### BFGs
	mesh.nBFaceGroup = 4
	mesh.allocate_bface_groups()
	for i in range(mesh.nBFaceGroup):
		BFG = mesh.BFaceGroups[i]
		if i == 0:
			BFG.Name = "x1"
			BFG.nBFace = nElem_y
		if i == 1:
			BFG.Name = "x2"
			BFG.nBFace = nElem_y
		if i == 2:
			BFG.Name = "y1"
			BFG.nBFace = nElem_x
		if i == 3:
			BFG.Name = "y2"
			BFG.nBFace = nElem_x
		BFG.allocate_bfaces()

	# x1
	BFG = mesh.BFaceGroups[0]
	n = 0
	for BF in BFG.BFaces:
		BF.Elem = nElem_x*n
		BF.face = 3
		n += 1
	# x2
	BFG = mesh.BFaceGroups[1]
	n = 0
	for BF in BFG.BFaces:
		BF.Elem = nElem_x*(n + 1) - 1
		BF.face = 1
		n += 1
	# y1
	BFG = mesh.BFaceGroups[2]
	n = 0
	for BF in BFG.BFaces:
		BF.Elem = n
		BF.face = 0
		n += 1
	# y2
	BFG = mesh.BFaceGroups[3]
	n = 0
	for BF in BFG.BFaces:
		BF.Elem = mesh.nElem - nElem_x + n
		BF.face = 2
		n += 1


	### IFaces
	mesh.nIFace = nElem_y*(nElem_x-1) + nElem_x*(nElem_y-1)
	mesh.allocate_ifaces()
	mesh.IFaces = [IFace() for i in range(mesh.nIFace)]
	# x direction
	n = 0
	for ny in range(nElem_y):
		for nx in range(nElem_x-1):
			IF = mesh.IFaces[n]
			IF.ElemL = nElem_x*ny + nx
			IF.faceL = 1
			IF.ElemR = nElem_x*ny + nx + 1
			IF.faceR = 3
			n += 1

	# y direction
	for nx in range(nElem_x):
		for ny in range(nElem_y-1):
			IF = mesh.IFaces[n]
			IF.ElemL = nElem_x*ny + nx
			IF.faceL = 2
			IF.ElemR = nElem_x*(ny + 1) + nx
			IF.faceR = 0
			n += 1

	mesh.allocate_helpers()
	mesh.fill_faces()

	return mesh


def split_quadrils_into_tris(mesh_old):
	nElem_old = mesh_old.nElem 
	nElem = nElem_old*2

	mesh = copy.deepcopy(mesh_old)

	mesh.SetParams(nElem=nElem, gbasis=Basis.LagrangeEqTri(1))

	def reorder_nodes(QOrder, nNodePerQuadril, nNodePerTri):
		nNodePerFace = QOrder + 1
		if nNodePerFace != np.sqrt(nNodePerQuadril):
			return ValueError
		quadril_nodes = np.arange(nNodePerQuadril)
		quadril_nodes.shape = nNodePerFace, nNodePerFace
		# faces 0 and 3 become faces 0 and 2 of tri1
		tri1_nodes = np.arange(nNodePerTri)
		# faces 1 and 2 become faces 2 and 0 of tri2
		tri2_nodes = np.copy(tri1_nodes)

		n = 0
		for j in range(nNodePerFace):
			tri1_nodes[n:n+nNodePerFace-j] = quadril_nodes[j,:nNodePerFace-j]
			if j == 0:
				tri2_nodes[n:n+nNodePerFace-j] = quadril_nodes[nNodePerFace-1,::-1]
			else:
				tri2_nodes[n:n+nNodePerFace-j] = quadril_nodes[nNodePerFace-(j+1),
						nNodePerFace-1:j-1:-1]
			n += nNodePerFace-j

		return tri1_nodes, tri2_nodes

	tri1_nodes, tri2_nodes = reorder_nodes(mesh.gorder, mesh_old.nNodePerElem, mesh.nNodePerElem)


	# Elems
	mesh.allocate_elem_to_nodes()
	for ielem in range(nElem_old):
		# First triangle
		mesh.Elem2Nodes[ielem] = mesh_old.Elem2Nodes[ielem, tri1_nodes]
		# Second triangle
		mesh.Elem2Nodes[ielem+nElem_old] = mesh_old.Elem2Nodes[ielem, tri2_nodes]


	old_to_new_face = np.array([2, 1, 2, 1])

	def modify_face_info(elem, face):
		if face == 1 or face == 2:
			elem += nElem_old
		face = old_to_new_face[face]
		return elem, face

	# BFGs
	for BFG in mesh.BFaceGroups:
		for BF in BFG.BFaces:
			BF.Elem, BF.face = modify_face_info(BF.Elem, BF.face)


	# Modify IFaces
	for IF in mesh.IFaces:
		IF.ElemL, IF.faceL = modify_face_info(IF.ElemL, IF.faceL)
		IF.ElemR, IF.faceR = modify_face_info(IF.ElemR, IF.faceR)

	# New IFaces
	mesh.nIFace += nElem_old
	for ielem in range(nElem_old):
		IF = IFace()
		IF.ElemL = ielem
		IF.faceL = 0
		IF.ElemR = ielem + nElem_old
		IF.faceR = 0
		mesh.IFaces.append(IF)

	mesh.allocate_faces()
	mesh.fill_faces()

	return mesh












