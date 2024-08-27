TimeStepping = {
	"FinalTime" : 0.2,
	"TimeStepSize" : 0.01,
	"TimeStepper" : "LSRK4",
}

Numerics = {
	"SolutionOrder" : 0,
	"SolutionBasis" : "LagrangeTri",
	"ElementQuadrature" : "Dunavant",
	"FaceQuadrature" : "GaussLegendre",
}

Mesh = {
	"ElementShape" : "Triangle",
	"NumElemsX" : 10,
	"NumElemsY" : 10,
	"xmin" : -1.,
	"xmax" : 1.,
	"ymin" : -1.,
	"ymax" : 1.,
	"PeriodicBoundariesX" : ["x2", "x1"],
	"PeriodicBoundariesY" : ["y2", "y1"],
}

Physics = {
	"Type" : "ElaAntiplain",
	"ConvFluxNumerical" : "LaxFriedrichs",
	"ShearModulus" : 1.,
	"Density" : 1.,
}

InitialCondition = {
	"Function" : "PlaneSine",
}

ExactSolution = InitialCondition.copy()

# d = {
# 		"BCType" : "Extrapolate",
# }

# BoundaryConditions = {
# 	"x1" : d,
# 	"x2" : d,
#     "y1" : d,
# 	"y2" : d,
# }

Output = {
	"AutoPostProcess" : True,
	"Verbose" : True,
}
