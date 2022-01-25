#!/usr/bin/env python3

import numpy as np
from dedalus import public as de
from dedalus.extras import flow_tools
import shutil
from glob import glob
import os.path as path
import time
import sys

import utils

def run(data_dir):
    
    # Read parameters from file
    params = utils.read_params(data_dir)
    
    ##################################################
    # Set up domain, fields and equations

    x_basis = de.Fourier('x',params["resX"],interval=(-params["Lx"]/2.,params["Lx"]/2.),dealias=2)   # Fourier basis in x
    y_basis = de.Fourier('y',params["resY"],interval=(-params["Ly"]/2.,params["Ly"]/2.),dealias=2)   # Fourier basis in y
    z_basis = de.Chebyshev('z',params["resZ"],interval=(0,params["Lz"]),dealias=2) # Chebyshev basis in z
    domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64, mesh=params["mesh"])  # Defining domain
    z = domain.grid(1, scales=1)                                   # accessing the z values

    # 3D Boussinesq hydrodynamics
    problem = de.IVP(domain,variables=['T','p','u','v','w','Tz','uz','vz','wz','ut','vt'])

    # Defining model parameters
    problem.parameters['Lx'] = params["resX"]
    problem.parameters['Ly'] = params["resY"]
    problem.parameters['Lz'] = params["resZ"]
    problem.parameters['Ra'] = params["Ra"]
    problem.parameters['Pr'] = params["Pr"]
    problem.parameters['Ta'] = params["Ta"]
    problem.parameters['Theta'] = params["Theta"]
    problem.parameters['X'] = params["Ra"]/params["Pr"]


    # # Defining d/dz of T, u, v and w for reducing our equations to first order
    # problem.add_equation("dz(u) - uz = 0")
    # problem.add_equation("dz(v) - vz = 0")
    # problem.add_equation("dz(w) - wz = 0")
    # problem.add_equation("dz(T) - Tz = 0")
    # problem.add_equation("dt(u) - ut = 0")
    # problem.add_equation("dt(v) - vt = 0")

    # # mass continuity
    # problem.add_equation("dx(u) + dy(v) + wz = 0")
    # # x-component of the momentum equation
    # problem.add_equation("dt(u) + dx(p) - (dx(dx(u)) + dy(dy(u)) + dz(uz)) - (Ta ** 0.5) * v * sin(Theta) = - (u * dx(u) + v * dy(u) + w * uz)")
    # # y-component of the momentum equation
    # problem.add_equation("dt(v) + dy(p) - (dx(dx(v)) + dy(dy(v)) + dz(vz)) + (Ta ** 0.5) * (u * sin(Theta) - w * cos(Theta)) = - (u * dx(v) + v * dy(v) + w * vz)")
    # # z-component of the momentum equation
    # problem.add_equation("dt(w) + dz(p) - (dx(dx(w)) + dy(dy(w)) + dz(wz)) - (Ta ** 0.5) * v * cos(Theta) - X * T = -(u * dx(w) + v * dy(w) + w * wz)")
    # # Temperature equation
    # problem.add_equation("Pr * dt(T) - (dx(dx(T)) + dy(dy(T)) + dz(Tz)) = - Pr * (u * dx(T) + v * dy(T) + w * Tz)")

    # problem.add_bc("left(dz(u)) = 0")           # free-slip boundary
    # problem.add_bc("right(dz(u)) = 0")          # free-slip boundary
    # problem.add_bc("left(dz(v)) = 0")           # free-slip boundary
    # problem.add_bc("right(dz(v)) = 0")          # free-slip boundary
    # problem.add_bc("left(w) = 0")            # Impermeable bottom boundary
    # problem.add_bc("right(w) = 0",condition="(nx != 0) or (ny != 0)")   # Impermeable top boundary
    # problem.add_bc("right(p) = 0",condition="(nx == 0) and (ny == 0)")   # Required for equations to be well-posed - see https://bit.ly/2nPVWIg for a related discussion
    # problem.add_bc("right(T) = 0")           # Fixed temperature at upper boundary
    # problem.add_bc("left(Tz) = -1")           # Fixed flux at bottom boundary, F = F_cond

    # Build solver
    solver = problem.build_solver(de.timesteppers.RK443)
    
    ##################################################
    # Initialise fields, either from previous sim or afresh
    
    if path.exists(path.join(data_dir, 'analysis.h5')):
        filepath = path.join(data_dir, 'analysis.h5')
        solver.load_state(filepath)
    else:
        print("Please run simulation and merge results first")
        exit(1)
            
    
    ##################################################
    # Prepare directory for simulation results
    
    analysis = solver.evaluator.add_file_handler(path.join(data_dir, "flattened/"), mode='overwrite')
    # Derivatives seem to be more accurate when calculated in dedalus, rather than in post
    analysis.add_task("integ(dx(u), 'y') / Ly", layout='g', name='u_dx')
    analysis.add_task("integ(dx(v), 'y') / Ly", layout='g', name='v_dx')
    analysis.add_task("integ(dx(w), 'y') / Ly", layout='g', name='w_dx')
    analysis.add_task("integ(dy(u), 'y') / Ly", layout='g', name='u_dy')
    analysis.add_task("integ(dy(v), 'y') / Ly", layout='g', name='v_dy')
    analysis.add_task("integ(dy(w), 'y') / Ly", layout='g', name='w_dy')
    analysis.add_task("integ(dz(u), 'y') / Ly", layout='g', name='u_dz')
    analysis.add_task("integ(dz(v), 'y') / Ly", layout='g', name='v_dz')
    analysis.add_task("integ(dz(w), 'y') / Ly", layout='g', name='w_dz')
    analysis.add_task("integ(dz(dz(u)), 'y') / Ly", layout='g', name='u_dz2')
    analysis.add_task("integ(dz(dz(v)), 'y') / Ly", layout='g', name='v_dz2')
    analysis.add_task("integ(dz(dz(w)), 'y') / Ly", layout='g', name='w_dz2')
    analysis.add_task("integ(ut, 'y') / Ly", layout='g', name='u_dt')
    analysis.add_task("integ(vt, 'y') / Ly", layout='g', name='v_dt')

    solver.evaluator.evaluate_handlers((analysis,))
    
    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please provide one argument: The file path to the directory to save the analysis in.")
        exit(1)
    data_dir = sys.argv[1]
    run(data_dir)
