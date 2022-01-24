#!/usr/bin/env python3

import numpy as np
from dedalus import public as de
from dedalus.extras import flow_tools
import shutil
from glob import glob
import os.path as path
from mpi4py import MPI
import time
import sys
from scipy import signal

import utils

def run(data_dir):
    
    # Read parameters from file
    params = utils.read_params(data_dir)

    print('Using the following parameters:')
    print(str(params))
    
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
    problem.parameters['X'] = Ra/Pr


    # Defining d/dz of T, u, v and w for reducing our equations to first order
    problem.add_equation("dz(u) - uz = 0")
    problem.add_equation("dz(v) - vz = 0")
    problem.add_equation("dz(w) - wz = 0")
    problem.add_equation("dz(T) - Tz = 0")
    problem.add_equation("dt(u) - ut = 0")
    problem.add_equation("dt(v) - vt = 0")

    # mass continuity
    problem.add_equation("dx(u) + dy(v) + wz = 0")
    # x-component of the momentum equation
    problem.add_equation("dt(u) + dx(p) - (dx(dx(u)) + dy(dy(u)) + dz(uz)) - (Ta ** 0.5) * v * sin(Theta) = - (u * dx(u) + v * dy(u) + w * uz)")
    # y-component of the momentum equation
    problem.add_equation("dt(v) + dy(p) - (dx(dx(v)) + dy(dy(v)) + dz(vz)) + (Ta ** 0.5) * (u * sin(Theta) - w * cos(Theta)) = - (u * dx(v) + v * dy(v) + w * vz)")
    # z-component of the momentum equation
    problem.add_equation("dt(w) + dz(p) - (dx(dx(w)) + dy(dy(w)) + dz(wz)) - (Ta ** 0.5) * v * cos(Theta) - X * T = -(u * dx(w) + v * dy(w) + w * wz)")
    # Temperature equation
    problem.add_equation("Pr * dt(T) - (dx(dx(T)) + dy(dy(T)) + dz(Tz)) = - Pr * (u * dx(T) + v * dy(T) + w * Tz)")

    problem.add_bc("left(dz(u)) = 0")           # free-slip boundary
    problem.add_bc("right(dz(u)) = 0")          # free-slip boundary
    problem.add_bc("left(dz(v)) = 0")           # free-slip boundary
    problem.add_bc("right(dz(v)) = 0")          # free-slip boundary
    problem.add_bc("left(w) = 0")            # Impermeable bottom boundary
    problem.add_bc("right(w) = 0",condition="(nx != 0) or (ny != 0)")   # Impermeable top boundary
    problem.add_bc("right(p) = 0",condition="(nx == 0) and (ny == 0)")   # Required for equations to be well-posed - see https://bit.ly/2nPVWIg for a related discussion
    problem.add_bc("right(T) = 0")           # Fixed temperature at upper boundary
    problem.add_bc("left(Tz) = -1")           # Fixed flux at bottom boundary, F = F_cond

    # Build solver
    solver = problem.build_solver(de.timesteppers.RK443)
    
    ##################################################
    # Initialise fields, either from previous sim or afresh
    
    if path.exists(path.join(data_dir, 'analysis.h5')):
        print("Analysis file already exists in the provided directory, will continue from where this simulation ended")
        filepath = path.join(data_dir, 'analysis.h5')
        solver.load_state(filepath)
    else:
        # We need to create a perturbation in the initial temperature field
        # Initial conditions
        z = domain.grid(1)
        T = solver.state['T']
        Tz = solver.state['Tz']

        # Random perturbations, initialized globally for same results in parallel
        gshape = domain.dist.grid_layout.global_shape(scales=1)
        slices = domain.dist.grid_layout.slices(scales=1)
        rand = np.random.RandomState(seed=42)
        noise = rand.standard_normal(gshape)[slices]

        # Linear background + perturbations damped at walls
        zb, zt = z_basis.interval
        pert =  1e-5 * noise * (zt - z) * (z - zb)
        T['g'] = pert
        T.differentiate('z', out=Tz)
            
    
    ##################################################
    # Prepare directory for simulation results
    
    print('Preparing analysis tasks...')
    # if path.exists(path.join(data_dir, 'analysis.h5')):
    #     shutil.move(filepath, path.join(data_dir, 'analysis_previous.h5'))
    analysis = solver.evaluator.add_file_handler(data_dir, sim_dt=params["timestep_analysis"], mode='overwrite')
    analysis.add_system(solver.state, layout='g')
    analysis.add_task("integ(integ(integ(0.5 * (u*u + v*v + w*w), 'x'), 'y'), 'z') / (Lx * Ly * Lz)", layout='g', name='E')
    analysis.add_task("T * w - Tz", layout='g', name='FluxHeat')
    analysis.add_task("integ(integ(T * w, 'x'), 'y') / (Lx * Ly)", layout='g', name='FluxHeatConv')
    analysis.add_task("integ(integ(-Tz, 'x'), 'y') / (Lx * Ly)", layout='g', name='FluxHeatCond')
    # Derivatives seem to be more accurate when calculated in dedalus, rather than in post
    analysis.add_task("dx(u)", layout='g', name='u_dx')
    analysis.add_task("dx(v)", layout='g', name='v_dx')
    analysis.add_task("dx(w)", layout='g', name='w_dx')
    analysis.add_task("dy(u)", layout='g', name='u_dy')
    analysis.add_task("dy(v)", layout='g', name='v_dy')
    analysis.add_task("dy(w)", layout='g', name='w_dy')
    analysis.add_task("dz(u)", layout='g', name='u_dz')
    analysis.add_task("dz(v)", layout='g', name='v_dz')
    analysis.add_task("dz(w)", layout='g', name='w_dz')
    analysis.add_task("dz(dz(u))", layout='g', name='u_dz2')
    analysis.add_task("dz(dz(v))", layout='g', name='v_dz2')
    analysis.add_task("dz(dz(w))", layout='g', name='w_dz2')
    analysis.add_task("ut", layout='g', name='u_dt')
    analysis.add_task("vt", layout='g', name='v_dt')
    
    ##################################################
    # Configure CFL to adjust timestep dynamically
    
    CFL = flow_tools.CFL(solver, initial_dt=params["timestep"], cadence=10, safety=0.5, max_change=1.5, min_change=0.5, max_dt=1e-4, threshold=0.05)
    CFL.add_velocities(('u', 'v', 'w'))
    flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
    
    ##################################################
    # Run the simulation
    
    solver.stop_sim_time = params["duration"]
    solver.stop_wall_time = np.inf
    solver.stop_iteration = np.inf
    print("Simulation start")
    sim_time_start = solver.sim_time
    t0 = time.time()
    dt = params["timestep"]
    # reset_averages()
    while solver.proceed:
        # Step the simulation forwards
        dt = CFL.compute_dt()
        dt = solver.step(dt)
        
        # Log the progress
        if (solver.iteration - 1) % 10 == 0:
            print(''.join([' '] * 200), end='\r')
            print(
                'Completed iteration {} (t = {:.3E}, dt = {:.3E}, {:.1f}%)'
                .format(solver.iteration, solver.sim_time, dt, 100 * (solver.sim_time - sim_time_start) / (params["duration"] - sim_time_start)),
            end='\r', flush=True)

    print(f'Simulation finished in {time.time() - t0} seconds')


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please provide one argument: The file path to the directory to save the analysis in.")
        exit(1)
    data_dir = sys.argv[1]
    run(data_dir)