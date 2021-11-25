#!/usr/bin/env python

import numpy as np
from dedalus import public as de
from dedalus.extras import flow_tools
import shutil
from glob import glob
import os.path as path
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

    x_basis = de.Fourier('x', params["resX"], interval=(0, params["Lx"]), dealias=2)
    y_basis = de.Fourier('y', params["resX"], interval=(0, params["Lx"]), dealias=2)
    z_basis = de.Chebyshev('z', params["resZ"], interval=(0, params["Lz"]), dealias=2)
    domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64, mesh=[4, 4])

    # u -- x component of flow vel
    # v -- y component of flow vel
    # w -- z component of flow vel
    # uz -- partial du/dz
    # vz -- partial dv/dz
    # wz -- partial dw/dz
    # T -- temperature
    # Tz -- partial dT/dz
    # p -- pressure
    problem = de.IVP(domain, variables=['u', 'v', 'w', 'uz', 'vz', 'wz', 'T', 'Tz', 'p'])#, *average_fields])
    problem.parameters['Ra'] = params["Ra"]
    problem.parameters['Pr'] = params["Pr"]
    problem.parameters['Ek'] = params["Ek"]
    problem.parameters['Theta'] = params["Theta"]
    problem.parameters['Lx'] = params["Lx"]
    problem.parameters['Lz'] = params["Lz"]

    # Nondimensionalised Boussinesq equations
    problem.add_equation("dt(u) + dx(p) - dx(dx(u)) - dy(dy(u)) - dz(uz) - v*sin(Theta)/Ek = -u*dx(u) - v*dy(u) - w*uz")
    problem.add_equation("dt(v) + dy(p) - dx(dx(v)) - dy(dy(v)) - dz(vz) - w*cos(Theta)/Ek + u*sin(Theta)/Ek = -u*dx(v) - v*dy(v) - w*vz")
    problem.add_equation("dt(w) + dz(p) - dx(dx(w)) - dy(dy(w)) - dz(wz) + v*cos(Theta)/Ek - Ra/Pr * T = -u*dx(w) - v*dy(w) - w*wz")

    # Convection-diffusion equation, governs evolution of temperature field
    problem.add_equation("dt(T) - (dx(dx(T)) + dy(dy(T)) + dz(Tz)) / Pr = -u*dx(T) - v*dy(T) - w*Tz")

    # Continuity equation
    problem.add_equation("dx(u) + dy(v) + wz = 0")

    # Substitutions for derivatives
    problem.add_equation("dz(u) - uz = 0")
    problem.add_equation("dz(v) - vz = 0")
    problem.add_equation("dz(w) - wz = 0")
    problem.add_equation("dz(T) - Tz = 0")

    # Boundary conditions
    problem.add_bc("left(Tz) = -1")
    # problem.add_bc("left(T) = 20")
    problem.add_bc("right(T) = 0")
    problem.add_bc("left(u) = 0")
    problem.add_bc("left(v) = 0")
    problem.add_bc("left(w) = 0")
    problem.add_bc("right(u) = 0")
    problem.add_bc("right(v) = 0")
    problem.add_bc("right(w) = 0", condition="(nx != 0) or (ny != 0)")
    problem.add_bc("right(p) = 0", condition="(nx == 0) and (ny == 0)")

    solver = problem.build_solver("RK222")
    
    
    ##################################################
    # Initialise fields, either from previous sim or afresh
    
    if path.exists(path.join(data_dir, 'snapshots', 'snapshots.h5')):
        print("Analysis file already exists in the provided directory, will continue from where this simulation ended")
        filepath = path.join(data_dir, 'snapshots', 'snapshots.h5')
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
    analysis = solver.evaluator.add_file_handler(path.join(data_dir, 'snapshots'), sim_dt=params["timestep_analysis"], mode='overwrite')
    analysis.add_system(solver.state, layout='g')
    analysis.add_task("integ(integ(integ(0.5 * (u*u + w*w), 'x'), 'y'), 'z') / (Lx * Lx * Lz)", layout='g', name='E')
    analysis.add_task("uz", layout='g', name='u_dz')
    analysis.add_task("vz", layout='g', name='v_dz')
    analysis.add_task("wz", layout='g', name='w_dz')
    analysis.add_task("dz(uz)", layout='g', name='u_dz2')
    analysis.add_task("dz(vz)", layout='g', name='v_dz2')
    analysis.add_task("dz(wz)", layout='g', name='w_dz2')

    averaged_y = solver.evaluator.add_file_handler(data_dir, sim_dt=params["timestep_analysis"], mode='overwrite')
    averaged_y.add_task("integ(u, 'y') / Lx", layout='g', name='u')
    averaged_y.add_task("integ(v, 'y') / Lx", layout='g', name='v')
    averaged_y.add_task("integ(w, 'y') / Lx", layout='g', name='w')
    averaged_y.add_task("integ(T, 'y') / Lx", layout='g', name='T')
    averaged_y.add_task("integ(uz, 'y') / Lx", layout='g', name='u_dz')
    averaged_y.add_task("integ(vz, 'y') / Lx", layout='g', name='v_dz')
    averaged_y.add_task("integ(wz, 'y') / Lx", layout='g', name='w_dz')
    averaged_y.add_task("integ(dz(uz), 'y') / Lx", layout='g', name='u_dz2')
    averaged_y.add_task("integ(dz(vz), 'y') / Lx", layout='g', name='v_dz2')
    averaged_y.add_task("integ(dz(wz), 'y') / Lx", layout='g', name='w_dz2')
    averaged_y.add_task("integ(Tz, 'y') / Lx", layout='g', name='Tz')

    
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

