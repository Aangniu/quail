TimeStepping = {
	"FinalTime" : 3.0,
	"TimeStepSize" : 0.01,
	"TimeStepper" : "LSRK4",
}

Numerics = {
	"SolutionOrder" : 1,
	"SolutionBasis" : "LagrangeTri",
	"ElementQuadrature" : "Dunavant",
	"FaceQuadrature" : "GaussLegendre",
}

Mesh = {
	"File" : "meshes/mesh_drFine.msh",
	# "PeriodicBoundariesX" : ["x2", "x1"],
	# "PeriodicBoundariesY" : ["y2", "y1"],
}

Physics = {
	"Type" : "ElaAntiplain",
	"ConvFluxNumerical" : "LaxFriedrichs",
	"ShearModulus" : 1.,
	"Density" : 1.,
}

InitialCondition = {
	"Function" : "Zeros",
}

ExactSolution = InitialCondition.copy()

SourceTerms = {
	"Source1" : { # Name of source term ("Source1") doesn't matter
		"Function" : "PointSource",
	},
}

d = {
		"BCType" : "Extrapolate",
}

BoundaryConditions = {
	"x1" : d,
	"x2" : d,
    "y1" : d,
	"y2" : d,
}

# BoundaryConditions = {
# 	"wall" : d,
# }

Output = {
    "Prefix": "output/Data_drFastIter0p1",
	"AutoPostProcess" : False,
	"Verbose" : True,
	"WriteInterval" : 50,
}
