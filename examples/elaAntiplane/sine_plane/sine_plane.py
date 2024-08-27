TimeStepping = {
	"FinalTime" : 1.0,
	"TimeStepSize" : 0.005,
	"TimeStepper" : "LSRK4",
}

Numerics = {
	"SolutionOrder" : 2,
	"SolutionBasis" : "LagrangeTri",
	"ElementQuadrature" : "Dunavant",
	"FaceQuadrature" : "GaussLegendre",
}

Mesh = {
	"ElementShape" : "Triangle",
	"NumElemsX" : 40,
	"NumElemsY" : 40,
	"xmin" : -1.,
	"xmax" : 1.,
	"ymin" : -1.,
	"ymax" : 1.,
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

Output = {
	"AutoPostProcess" : True,
	"Verbose" : True,
	"WriteInterval" : 10,
}
