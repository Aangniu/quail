import sys; sys.path.append('../../../../src'); sys.path.append('./src')
import numpy as np
import code
import MeshCommon
import General
import matplotlib as mpl
from matplotlib import pyplot as plt
import Plot
from helper import *


### Parameters
Orders = np.arange(1,6)
nOrders = len(Orders)
basis = General.BasisType["SegLagrange"]
nL = 51 # number of wavenumbers


### Mesh
# Note: dummy mesh
mesh = MeshCommon.Mesh1D(Uniform=True, nElem=2, xmin=-1., xmax=1., Periodic=True)
h = mesh.Coords[1,0] - mesh.Coords[0,0]


### Allocate
L = np.linspace(0., np.pi, nL)
Omega_r_all = np.zeros(nL)
Omega_i_all = np.zeros_like(Omega_r_all)


### Prepare plotting
Plot.PreparePlot()
figR = plt.figure(0)
figI = plt.figure(1)


### Hard-coded tables that determine which mode to use at different intervals
## Specific to nL = 51 and for p = 1,2,...,5
ModeIntervals = [None]*10
ModesToUse = [None]*10
ModeIntervals[1] = np.array([0.499,1.])
ModesToUse[1] = np.array([0,1])
ModeIntervals[2] = np.array([0.338,0.677,1.])
ModesToUse[2] = np.array([2,1,0])
ModeIntervals[3] = np.array([0.001, 0.257, 0.494, 0.518, 0.734, 0.751, 0.774, 1.])
ModesToUse[3] = np.array([3,2,3,2,1,0,1,0])
ModeIntervals[4] = np.array([0.212, 0.317, 0.419, 0.498, 0.596, 0.798, 1.])
ModesToUse[4] = np.array([4,3,2,3,2,1,0])
ModeIntervals[5] = np.array([0.0376, 0.178, 0.239, 0.338, 0.378, 0.438, 0.5, 0.637, 0.675, 0.697, 0.837, 1.])
ModesToUse[5] = np.array([4,5,4,3,5,4,3,2,1,2,1,0])


### Loop through each order
for p in Orders:
	MMinv, SM, PhiLeft, PhiRight, nn = CalculateBasisAndMatrices(mesh, basis, p)

	Omega_r_all *= 0.; Omega_i_all *= 0.
	for i in range(nL):
		Omega_r, Omega_i = GetEigValues(MMinv, SM, PhiLeft, PhiRight, L[i], p, h)

		# Cross-check with hard-coded tables
		modes = ModesToUse[p]
		intervals = ModeIntervals[p]
		if modes is None or intervals is None:
			raise Exception("Order not supported")

		Lp = L[i]/np.pi
		for j in range(len(intervals)):
			if Lp <= intervals[j]:
				break
		mode = modes[j]

		# Store
		Omega_r_all[i] = Omega_r[mode]
		Omega_i_all[i] = Omega_i[mode]

	## Plot
	# Dispersion
	plt.figure(figR.number)
	plt.plot(L/np.pi, Omega_r_all/nn, '--', label="$p = %d$" % (p))
	# Dissipation
	plt.figure(figI.number)
	plt.plot(L/np.pi, Omega_i_all/nn, '--', label="$p = %d$" % (p))


### Finalize plots
# Dispersion
plt.figure(figR.number)
plt.plot(np.array([L[0], L[-1]])/np.pi, np.array([L[0], L[-1]]), 'k:', label="Exact")
plt.xlabel("$L/\\pi$")
plt.ylabel("$\\Omega_r/N_p$")
plt.legend(loc="best")

# Dissipation
plt.figure(figI.number)
plt.plot(np.array([L[0], L[-1]])/np.pi, np.array([0.,0.]), 'k:', label="Exact")
plt.xlabel("$L/\\pi$")
plt.ylabel("$\\Omega_i/N_p$")
plt.legend(loc="best")

plt.show()

