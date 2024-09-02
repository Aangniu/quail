import solver.DG as solver_DG

class DRFaceHelpers(solver_DG.ElemHelpers):
	'''
	The DRFaceHelpers class contains the methods and attributes that
	are accessed prior to the main solver temporal loop. They are used to
	precompute attributes for the rupture faces in the domain.

	DRFaceHelpers inherits attributes from the ElemHelpers parent
	class. See ElemHelpers class for additional comments of methods.

	Attributes:
	-----------
	quad_pts: numpy array
		coordinates for the quadrature point evaluations
	quad_wts: numpy array
		values for the weights of each quadrature point
	faces_to_basisL: numpy array
		basis values evaluated at quadrature points of each face for
		left element
	faces_to_basisR: numpy array
		basis values evaluated at quadrature points of each face for
		right element
	faces_to_basis_ref_gradL: numpy array
		gradient of basis values evaluated at quadrature points of each face
		for left element
	faces_to_basis_ref_gradR: numpy array
		gradient of basis values evaluated at quadrature points of each face
		for right element
	normals_int_faces: numpy array
		normal vector array for each interior face
	UqL: numpy array
		solution vector evaluated at the face quadrature points for left
		element
	UqR: numpy array
		solution vector evaluated at the face quadrature points for right
		element
	Fq: numpy array
		flux vector evaluated at the face quadrature points
	elemL_IDs: numpy array
		element IDs to the left of each interior face
	elemR_IDs: numpy array
		element IDs to the right of each interior face
	faceL_IDs: numpy array
		face IDs to the left of each interior face
	faceR_IDs: numpy array
		face IDs to the right of each interior face
	ijacL_elems: numpy array
		stores the evaluated inverse of the geometric Jacobian for each
		left element
	ijacR_elems: numpy array
		stores the evaluated inverse of the geometric Jacobian for each
		right element


	Methods:
	--------
	get_gaussian_quadrature
		precomputes the quadrature points and weights for the given
		quadrature type
	get_basis_and_geom_data
		precomputes the face's basis function, its gradients,
		and normals
	alloc_other_arrays
		allocate the solution and flux vectors that are evaluated
		at the quadrature points
	compute_helpers
		call the functions to precompute the necessary helper data
	'''
	def __init__(self):
		self.quad_pts = np.zeros(0)
		self.quad_wts = np.zeros(0)
		self.faces_to_basisL = np.zeros(0)
		self.faces_to_basisR = np.zeros(0)
		self.faces_to_basis_ref_gradL = np.zeros(0)
		self.faces_to_basis_ref_gradR = np.zeros(0)
		self.normals_int_faces = np.zeros(0)
		self.UqL = np.zeros(0)
		self.UqR = np.zeros(0)
		self.Fq = np.zeros(0)
		self.elemL_IDs = np.empty(0, dtype=int)
		self.elemR_IDs = np.empty(0, dtype=int)
		self.faceL_IDs = np.empty(0, dtype=int)
		self.faceR_IDs = np.empty(0, dtype=int)
		self.ijacL_elems = np.zeros(0)
		self.ijacR_elems = np.zeros(0)

	def get_gaussian_quadrature(self, mesh, physics, basis, order):
		'''
		Precomputes the quadrature points and weights given the computed
		quadrature order

		Inputs:
		-------
			mesh: mesh object
			physics: physics object
			basis: basis object
			order: solution order

		Outputs:
		--------
			self.quad_pts: precomputed quadrature points [nq, ndims]
			self.quad_wts: precomputed quadrature weights [nq, 1]
		'''
		gbasis = mesh.gbasis
		quad_order = gbasis.FACE_SHAPE.get_quadrature_order(mesh,
				order, physics=physics)
		self.quad_pts, self.quad_wts = \
				basis.FACE_SHAPE.get_quadrature_data(quad_order)

	def get_basis_and_geom_data(self, mesh, basis, order):
		'''
		Precomputes the basis and geometric data for each interior face

		Inputs:
		-------
			mesh: mesh object
			basis: basis object
			order: solution order

		Outputs:
		--------
			self.faces_to_basisL: basis values evaluated at quadrature
				points of each face for left element
				[nfaces_per_elem, nq, nb]
			self.faces_to_basisR: basis values evaluated at quadrature
				points of each face for right element
				[nfaces_per_elem, nq, nb]
			self.faces_to_basis_ref_gradL: gradient of basis values
				evaluated at quadrature points of each face for left element
				[nfaces_per_elem, nq, nb, ndims]
			self.faces_to_basis_ref_gradR: gradient of basis values
				evaluated at quadrature points of each face for right element
				[nfaces_per_elem, nq, nb, ndims]
			self.normals_int_faces: precomputed normal vectors at each
				interior face [num_interior_faces, nq, ndims]
			self.ijacL_elems: stores the evaluated inverse of the geometric
				Jacobian for each left element
				[num_interior_faces, nq, ndims, ndims]
			self.ijacR_elems: stores the evaluated inverse of the geometric
				Jacobian for each right element
				[num_interior_faces, nq, ndims, ndims]
			self.face_lengths: stores the precomputed length of each face
				[num_interior_faces, 1]

		Note(s):
		--------
			We separate ndims_basis and ndims to allow for basis
			and mesh to have different number of dimensions
			(ex: when using a space-time basis function and
			only a spatial mesh)
		'''
		ndims_basis = basis.NDIMS
		ndims = mesh.ndims

		# unpack
		quad_pts = self.quad_pts
		quad_wts = self.quad_wts
		nq = quad_pts.shape[0]
		nb = basis.nb
		nfaces_per_elem = basis.NFACES
		# TODO: count the number of rupture faces
		nfaces = mesh.num_interior_faces

		# Allocate
		self.faces_to_basisL = np.zeros([nfaces_per_elem, nq, nb])
		self.faces_to_basisR = np.zeros([nfaces_per_elem, nq, nb])
		self.faces_to_basis_ref_gradL = np.zeros([nfaces_per_elem,
				nq, nb, ndims_basis])
		self.faces_to_basis_ref_gradR = np.zeros([nfaces_per_elem,
				nq, nb, ndims_basis])
		self.ijacL_elems = np.zeros([nfaces, nq, ndims, ndims])
		self.ijacR_elems = np.zeros([nfaces, nq, ndims, ndims])
		self.normals_int_faces = np.zeros([mesh.num_interior_faces, nq,
				ndims])
		djac_faces = np.zeros([mesh.num_interior_faces, nq])

		# Get values on each face (from both left and right perspectives)
		# for both the basis and the reference gradient of the basis
		for face_ID in range(nfaces_per_elem):
			# Left
			basis.get_basis_face_val_grads(mesh, face_ID, quad_pts,
					get_val=True, get_ref_grad=True)
			self.faces_to_basisL[face_ID] = basis.basis_val
			self.faces_to_basis_ref_gradL[face_ID] = basis.basis_ref_grad

			# Right
			basis.get_basis_face_val_grads(mesh, face_ID, quad_pts[::-1],
					get_val=True, get_ref_grad=True)
			self.faces_to_basisR[face_ID] = basis.basis_val
			self.faces_to_basis_ref_gradR[face_ID] = basis.basis_ref_grad

		# Normals
		i = 0
		for interior_face in mesh.interior_faces:
			normals = mesh.gbasis.calculate_normals(mesh,
					interior_face.elemL_ID, interior_face.faceL_ID, quad_pts)
			self.normals_int_faces[i] = normals

			# Left state
			# Convert from face ref space to element ref space
			elem_pts = basis.get_elem_ref_from_face_ref(
					interior_face.faceL_ID, quad_pts)

			_, _, ijacL = basis_tools.element_jacobian(mesh,
					interior_face.elemL_ID, elem_pts, get_djac=False,
					get_jac=False, get_ijac=True)

			# Right state
			# Convert from face ref space to element ref space
			elem_pts = basis.get_elem_ref_from_face_ref(
					interior_face.faceR_ID, quad_pts[::-1])
			_, _, ijacR = basis_tools.element_jacobian(mesh,
					interior_face.elemR_ID, elem_pts, get_djac=False,
					get_jac=False, get_ijac=True)

			# Store
			self.ijacL_elems[i] = ijacL
			self.ijacR_elems[i] = ijacR

			# Used for face_length calculations
			djac_faces[i] = np.linalg.norm(normals, axis=1)
			i += 1

		self.face_lengths = mesh_tools.get_face_lengths(djac_faces, quad_wts)

	def alloc_other_arrays(self, physics, basis, order):
		'''
		Allocates the solution and flux vectors that are evaluated
		at the quadrature points

		Inputs:
		-------
			physics: physics object
			basis: basis object
			order: solution order
		'''
		quad_pts = self.quad_pts
		nq = quad_pts.shape[0]
		ns = physics.NUM_STATE_VARS

		self.UqL = np.zeros([nq, ns])
		self.UqR = np.zeros([nq, ns])
		self.Fq = np.zeros([nq, ns])

	def store_neighbor_info(self, mesh):
		'''
		Store the element and face IDs on the left and right of each face.

		Inputs:
		-------
			mesh: mesh object

		Outputs:
		--------
			self.elemL_IDs: Element IDs to the left of each interior face
				[num_interior_faces]
			self.elemR_IDs: Element IDs to the right of each interior face
				[num_interior_faces]
			self.faceL_IDs: Face IDs to the left of each interior face
				[num_interior_faces]
			self.faceR_IDs: Face IDs to the right of each interior face
				[num_interior_faces]
		'''
		self.elemL_IDs = np.empty(mesh.num_interior_faces, dtype=int)
		self.elemR_IDs = np.empty(mesh.num_interior_faces, dtype=int)
		self.faceL_IDs = np.empty(mesh.num_interior_faces, dtype=int)
		self.faceR_IDs = np.empty(mesh.num_interior_faces, dtype=int)
		for face_ID in range(mesh.num_interior_faces):
			int_face = mesh.interior_faces[face_ID]
			self.elemL_IDs[face_ID] = int_face.elemL_ID
			self.elemR_IDs[face_ID] = int_face.elemR_ID
			self.faceL_IDs[face_ID] = int_face.faceL_ID
			self.faceR_IDs[face_ID] = int_face.faceR_ID

	def compute_helpers(self, mesh, physics, basis, order):
		self.get_gaussian_quadrature(mesh, physics, basis, order)
		self.get_basis_and_geom_data(mesh, basis, order)
		self.alloc_other_arrays(physics, basis, order)
		self.store_neighbor_info(mesh)