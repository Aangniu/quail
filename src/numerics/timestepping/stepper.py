from abc import ABC, abstractmethod
import code
import numpy as np 
from scipy.optimize import fsolve, root

from data import ArrayList
from general import StepperType, ODESolverType
from solver.tools import mult_inv_mass_matrix
import numerics.basis.tools as basis_tools
import numerics.timestepping.tools as stepper_tools
import numerics.timestepping.ode as ode
import solver.tools as solver_tools

class StepperBase(ABC):
	def __init__(self, U):
		self.R = np.zeros_like(U)
		self.dt = 0.
		self.numtimesteps = 0
		self.get_time_step = None
		self.balance_const = None
	def __repr__(self):
		return '{self.__class__.__name__}(TimeStep={self.dt})'.format(self=self)
	@abstractmethod
	def TakeTimeStep(self, solver):
		pass

class FE(StepperBase):

	def TakeTimeStep(self, solver):
		EqnSet = solver.EqnSet
		DataSet = solver.DataSet
		mesh = solver.mesh
		U = EqnSet.U

		R = self.R 
		R = solver.calculate_residual(U, R)
		dU = mult_inv_mass_matrix(mesh, solver, self.dt, R)
		U += dU

		solver.apply_limiter(U)

		return R


class RK4(StepperBase):

	def TakeTimeStep(self, solver):
		EqnSet = solver.EqnSet
		DataSet = solver.DataSet
		mesh = solver.mesh
		U = EqnSet.U

		R = self.R

		# first stage
		R = solver.calculate_residual(U, R)
		dU1 = mult_inv_mass_matrix(mesh, solver, self.dt, R)
		Utemp = U + 0.5*dU1
		solver.apply_limiter(Utemp)

		# second stage
		solver.Time += self.dt/2.
		R = solver.calculate_residual(Utemp, R)
		dU2 = mult_inv_mass_matrix(mesh, solver, self.dt, R)
		Utemp = U + 0.5*dU2
		solver.apply_limiter(Utemp)

		# third stage
		R = solver.calculate_residual(Utemp, R)
		dU3 = mult_inv_mass_matrix(mesh, solver, self.dt, R)
		Utemp = U + dU3
		solver.apply_limiter(Utemp)

		# fourth stage
		solver.Time += self.dt/2.
		R = solver.calculate_residual(Utemp, R)
		dU4 = mult_inv_mass_matrix(mesh, solver, self.dt, R)
		dU = 1./6.*(dU1 + 2.*dU2 + 2.*dU3 + dU4)
		U += dU
		solver.apply_limiter(U)

		return R


class LSRK4(StepperBase):
	# Low-storage RK4
	def __init__(self, U):
		super().__init__(U)
		self.rk4a = np.array([            0.0, \
		    -567301805773.0/1357537059087.0, \
		    -2404267990393.0/2016746695238.0, \
		    -3550918686646.0/2091501179385.0, \
		    -1275806237668.0/842570457699.0])
		self.rk4b = np.array([ 1432997174477.0/9575080441755.0, \
		    5161836677717.0/13612068292357.0, \
		    1720146321549.0/2090206949498.0, \
		    3134564353537.0/4481467310338.0, \
		    2277821191437.0/14882151754819.0])
		self.rk4c = np.array([             0.0, \
		    1432997174477.0/9575080441755.0, \
		    2526269341429.0/6820363962896.0, \
		    2006345519317.0/3224310063776.0, \
		    2802321613138.0/2924317926251.0])
		self.nstages = 5
		self.dU = np.zeros_like(U)

	def TakeTimeStep(self, solver):

		EqnSet = solver.EqnSet
		DataSet = solver.DataSet
		mesh = solver.mesh
		U = EqnSet.U

		R = self.R
		dU = self.dU

		Time = solver.Time
		for INTRK in range(self.nstages):
			solver.Time = Time + self.rk4c[INTRK]*self.dt
			R = solver.calculate_residual(U, R)
			dUtemp = mult_inv_mass_matrix(mesh, solver, self.dt, R)
			dU *= self.rk4a[INTRK]
			dU += dUtemp
			U += self.rk4b[INTRK]*dU
			solver.apply_limiter(U)

		return R

class SSPRK3(StepperBase):
	# Low-storage SSPRK3 with 5 stages (as written in Spiteri. 2002)
	def __init__(self, U):
		super().__init__(U)
		self.ssprk3a = np.array([	0.0, \
			-2.60810978953486, \
			-0.08977353434746, \
			-0.60081019321053, \
			-0.72939715170280])
		self.ssprk3b = np.array([ 0.67892607116139, \
			0.20654657933371, \
			0.27959340290485, \
			0.31738259840613, \
			0.30319904778284])
		self.nstages = 5
		self.dU = np.zeros_like(U)

	def TakeTimeStep(self, solver):
		EqnSet = solver.EqnSet
		DataSet = solver.DataSet
		mesh = solver.mesh
		U = EqnSet.U

		R = self.R
		dU = self.dU

		Time = solver.Time
		for INTRK in range(self.nstages):
			solver.Time = Time + self.dt
			R = solver.calculate_residual(U, R)
			dUtemp = mult_inv_mass_matrix(mesh, solver, self.dt, R)
			dU *= self.ssprk3a[INTRK]
			dU += dUtemp
			U += self.ssprk3b[INTRK]*dU
			solver.apply_limiter(U)
		return R	


class ADER(StepperBase):
	
	def TakeTimeStep(self, solver):
		EqnSet = solver.EqnSet
		DataSet = solver.DataSet
		mesh = solver.mesh
		W = EqnSet.U
		Up = EqnSet.Up

		R = self.R

		# Prediction Step (Non-linear Case)
		Up = solver.calculate_predictor_step(self.dt, W, Up)

		# Correction Step
		R = solver.calculate_residual(Up, R)

		dU = mult_inv_mass_matrix(mesh, solver, self.dt/2., R)

		W += dU
		solver.apply_limiter(W)
		return R

class Strang(StepperBase, ode.ODESolvers):

	def set_split_schemes(self, explicit, implicit, U):
		param = {"TimeScheme":explicit}
		self.explicit = stepper_tools.set_stepper(param, U)

		# self.implicit = stepper_tools.set_stepper(param, U)
		if ODESolverType[implicit] == ODESolverType.BDF1:
			self.implicit = ode.ODESolvers.BDF1(U)
		elif ODESolverType[implicit] == ODESolverType.Trapezoidal:
			self.implicit = ode.ODESolvers.Trapezoidal(U)
		else:
			raise NotImplementedError("Time scheme not supported")

	def TakeTimeStep(self, solver):

		EqnSet = solver.EqnSet
		DataSet = solver.DataSet
		mesh  = solver.mesh
		U = EqnSet.U

		explicit = self.explicit
		explicit.dt = self.dt/2.
		implicit = self.implicit
		implicit.dt = self.dt

		#First: take the half-step for the inviscid flux only
		solver.Params["SourceSwitch"] = False
		R1 = explicit.TakeTimeStep(solver)

		#Second: take the implicit full step for the source term.
		solver.Params["SourceSwitch"] = True
		solver.Params["ConvFluxSwitch"] = False

		R2 = implicit.TakeTimeStep(solver)

		#Third: take the second half-step for the inviscid flux only.
		solver.Params["SourceSwitch"] = False
		solver.Params["ConvFluxSwitch"] = True
		R3 = explicit.TakeTimeStep(solver)

		return R3

class Simpler(Strang):

	def TakeTimeStep(self, solver):

		EqnSet = solver.EqnSet
		DataSet = solver.DataSet
		mesh  = solver.mesh
		U = EqnSet.U

		explicit = self.explicit
		explicit.dt = self.dt/2.
		implicit = self.implicit
		implicit.dt = self.dt
		
		solver.Params["SourceSwitch"] = False
		R = self.R 

		self.balance_const = None
		balance_const = -1.*solver.calculate_residual(U, R)
		self.balance_const = -1.*balance_const

		#Second: take the implicit full step for the source term.
		solver.Params["SourceSwitch"] = True
		solver.Params["ConvFluxSwitch"] = False
		R2 = implicit.TakeTimeStep(solver)

		#Third: take the second half-step for the inviscid flux only.
		solver.Params["SourceSwitch"] = False
		solver.Params["ConvFluxSwitch"] = True
		self.balance_const = balance_const
		R3 = explicit.TakeTimeStep(solver)

		return R3


