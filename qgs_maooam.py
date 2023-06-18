#!/usr/bin/env python
# coding: utf-8

# ## Coupled ocean-atmosphere model version

# This model version is a 2-layer channel QG atmosphere truncated at wavenumber 2 coupled, both by friction
# and heat exchange, to a shallow water ocean with 8 modes.
#
# More detail can be found in the articles:
#
# * Vannitsem, S., Demaeyer, J., De Cruz, L., & Ghil, M. (2015). Low-frequency variability and heat
#   transport in a low-order nonlinear coupled ocean–atmosphere model. Physica D: Nonlinear Phenomena, 309, 71-85.
# * De Cruz, L., Demaeyer, J., and Vannitsem, S.: The Modular Arbitrary-Order Ocean-Atmosphere Model:
#   MAOOAM v1.0, Geosci. Model Dev., 9, 2793–2808, 2016.
#


# ## Modules import
import numpy as np
import sys
import time
from multiprocessing import freeze_support, get_start_method

# Importing the model's modules
from qgs.params.params import QgParams
from qgs.integrators.integrator import RungeKuttaIntegrator
from qgs.functions.tendencies import create_tendencies

# Initializing the random number generator (for reproducibility). -- Disable if needed.
np.random.seed(21217)

if __name__ == "__main__":

    if get_start_method() == "spawn":
        freeze_support()

    print_parameters = True


    def print_progress(p):
        sys.stdout.write('Progress {:.2%} \r'.format(p))
        sys.stdout.flush()


    class Bcolors:
        """to color the instructions in the console"""
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'


    print("\n" + Bcolors.HEADER + Bcolors.BOLD + "Model qgs v0.2.8 (Atmosphere + ocean (MAOOAM) configuration)" + Bcolors.ENDC)
    print(Bcolors.HEADER + "============================================================" + Bcolors.ENDC + "\n")
    print(Bcolors.OKBLUE + "Initialization ..." + Bcolors.ENDC)
    # ## Systems definition

    # General parameters

    # Time parameters
    dt = 0.1
    # Saving the model state n steps
    write_steps = 100
    # transient time to attractor
    transient_time = 3.e6
    # integration time on the attractor
    integration_time = 5.e5
    # file where to write the output
    filename = "evol_fields.dat"
    T = time.process_time()

    # Setting some model parameters
    # Model parameters instantiation with default specs
    model_parameters = QgParams()
    # Mode truncation at the wavenumber 2 in both x and y spatial coordinate
    model_parameters.set_atmospheric_channel_fourier_modes(2, 2)
    # Mode truncation at the wavenumber 2 in the x and at the
    # wavenumber 4 in the y spatial coordinates for the ocean
    model_parameters.set_oceanic_basin_fourier_modes(2, 4)

    # Setting MAOOAM parameters according to the publication linked above
    model_parameters.set_params({'kd': 0.0290, 'kdp': 0.0290, 'n': 1.5, 'r': 1.e-7,
                                 'h': 136.5, 'd': 1.1e-7})
    model_parameters.atemperature_params.set_params({'eps': 0.7, 'T0': 289.3, 'hlambda': 15.06, })
    model_parameters.gotemperature_params.set_params({'gamma': 5.6e8, 'T0': 301.46})

    model_parameters.atemperature_params.set_insolation(103.3333, 0)
    model_parameters.gotemperature_params.set_insolation(310., 0)

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
    ic = np.random.rand(model_parameters.ndim)*0.01
    # Integrate over a transient time to obtain an initial condition on the attractors
    print(Bcolors.OKBLUE + "Starting a transient time integration..." + Bcolors.ENDC)
    ws = 10000
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

    print(Bcolors.OKBLUE + "Starting the time evolution ..." + Bcolors.ENDC)
    while total_time < integration_time:
        integrator.integrate(0., write_steps * dt, dt, ic=y, write_steps=0)
        t, y = integrator.get_trajectories()
        total_time += t
        ty = np.insert(y, 0, total_time)
        traj = np.concatenate((traj, ty[np.newaxis, ...]))
        if total_time/integration_time*100 % 0.1 < t_up:
            print_progress(total_time/integration_time)

    print(Bcolors.OKGREEN + "Evolution finished, writing to file " + filename + Bcolors.ENDC)

    np.savetxt(filename, traj)

    print(Bcolors.OKGREEN + "Time clock :" + Bcolors.ENDC)
    print(str(time.process_time()-T)+' seconds')
