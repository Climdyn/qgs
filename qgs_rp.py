#!/usr/bin/env python
# coding: utf-8

# ## Reinhold and Pierrehumbert 1982 model version

# This model version is a simple 2-layer channel QG atmosphere truncated at wavenumber 2 on a beta-plane with
# a simple orography (a montain and a valley).
# 
# More detail can be found in the articles:
# 
# * Reinhold, B. B., & Pierrehumbert, R. T. (1982). Dynamics of weather regimes: Quasi-stationary waves and blocking.
#   Monthly Weather Review, 110(9), 1105-1145.
# * Cehelsky, P., & Tung, K. K. (1987). Theories of multiple equilibria and weather regimes—A critical reexamination.
#   Part II: Baroclinic two-layer models. Journal of the atmospheric sciences, 44(21), 3282-3303.


# ## Modules import
import numpy as np
import sys
import time

# Importing the model's modules
from params.params import QgParams
from integrators.integrator import RungeKuttaIntegrator
from functions.tendencies import create_tendencies

# Initializing the random number generator (for reproducibility). -- Disable if needed.
np.random.seed(21217)

print_parameters = True

def print_progress(p):
    sys.stdout.write('Progress {:.2%} \r'.format(p))
    sys.stdout.flush()


class bcolors:
    """to color the instructions in the console"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


print("\n" + bcolors.HEADER + bcolors.BOLD + "Model qgs (Atmosphere + orography configuration)" + bcolors.ENDC)
print(bcolors.HEADER + "================================================" + bcolors.ENDC + "\n")
print(bcolors.OKBLUE + "Initialization ..." + bcolors.ENDC)
# ## Systems definition

# General parameters

# Time parameters
dt = 0.1
# Saving the model state n steps
write_steps = 5
# transient time to attractor
transient_time = 1.e5
# integration time on the attractor
integration_time = 1.e4
# file where to write the output
filename = "evol_fields.dat"
T = time.process_time()

# Setting some model parameters
# Model parameters instantiation with some non-default specs
model_parameters = QgParams({'phi0_npi': np.deg2rad(50.)/np.pi, 'hd': 0.1})
# Mode truncation at the wavenumber 2 in both x and y spatial coordinate
model_parameters.set_atmospheric_modes(2, 2)

# Changing (increasing) the orography depth and the meridional temperature gradient
model_parameters.scale_params.hk[1] = 0.2                
model_parameters.atemperature_params.thetas[0] = 0.2

if print_parameters:
    print("")
    # Printing the model's parameters
    model_parameters.print_params()

# Creating the tendencies functions
f, Df = create_tendencies(model_parameters)

# ## Time integration
# Defining an integrator
integrator = RungeKuttaIntegrator()
integrator.set_func(f)

# Start on a random initial condition
ic = np.random.rand(model_parameters.ndim)*0.1
# Integrate over a transient time to obtain an initial condition on the attractors
print(bcolors.OKBLUE + "Starting a transient time integration..." + bcolors.ENDC)
ws = 1000
y = ic
total_time = 0.
t_up = ws * dt / integration_time * 100
while total_time < transient_time:
    integrator.integrate(0., ws * dt, dt, ic=y, write_steps=0)
    t, y = integrator.get_trajectories()
    total_time += t
    if total_time/transient_time * 100 % 0.1 < t_up:
        print_progress(total_time/transient_time)

# Now integrate to obtain a trajectory on the attractor
total_time = 0.
traj = np.insert(y, 0, total_time)
traj = traj[np.newaxis, ...]
t_up = write_steps * dt / integration_time * 100

print(bcolors.OKBLUE + "Starting the time evolution ..." + bcolors.ENDC)
while total_time < integration_time:
    integrator.integrate(0., write_steps * dt, dt, ic=y, write_steps=0)
    t, y = integrator.get_trajectories()
    total_time += t
    ty = np.insert(y, 0, total_time)
    traj = np.concatenate((traj, ty[np.newaxis, ...]))
    if total_time/integration_time*100 % 0.1 < t_up:
        print_progress(total_time/integration_time)

print(bcolors.OKGREEN + "Evolution finished, writing to file " + filename + bcolors.ENDC)

np.savetxt(filename, traj)

print(bcolors.OKGREEN + "Time clock :" + bcolors.ENDC)
print(str(time.process_time()-T)+' seconds')

