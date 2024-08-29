TimeStepping = {
	"FinalTime" : 10.0,
	"TimeStepSize" : 0.01,
	"TimeStepper" : "LSRK4",
}

Numerics = {
	"SolutionOrder" : 3,
	"SolutionBasis" : "LagrangeTri",
	"ElementQuadrature" : "Dunavant",
	"FaceQuadrature" : "GaussLegendre",
}

Mesh = {
	"ElementShape" : "Triangle",
	"NumElemsX" : 50,
	"NumElemsY" : 50,
	"xmin" : -5.,
	"xmax" : 5.,
	"ymin" : -5.,
	"ymax" : 5.,
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
    "Prefix": "output/Data",
	"AutoPostProcess" : True,
	"Verbose" : True,
	"WriteInterval" : 50,
}
