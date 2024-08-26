TimeStepping = {
	"FinalTime" : 2.0,
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
	"NumElemsX" : 20,
	"NumElemsY" : 20,
	"xmin" : -1.,
	"xmax" : 1.,
	"ymin" : -1.,
	"ymax" : 1.,
	"PeriodicBoundariesX" : ["x2", "x1"],
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

d = {
		"BCType" : "Extrapolate",
}

BoundaryConditions = {
	"y1" : d,
	"y2" : d,
}

Output = {
	"AutoPostProcess" : True,
	"Verbose" : True,
}
