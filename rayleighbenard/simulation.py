import h5py
import numpy as np
from dedalus import public as de
from dedalus.tools import post
from dedalus.extras import flow_tools
import shutil
import json
from glob import glob
import os.path as path
import time
import sys

import rayleighbenard.utils as utils

def run(data_dir):
    
    # Read parameters from file
    params = utils.read_params(data_dir)
    
    
    ##################################################
    # Set up domain, fields and equations

    x_basis = de.Fourier('x', params["resX"], interval=(0, params["Lx"]), dealias=2)
    z_basis = de.Chebyshev('z', params["resZ"], interval=(0, params["Lz"]), dealias=2)
    domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

    # u -- x component of flow vel
    # v -- y component of flow vel
    # w -- z component of flow vel
    # uz -- partial du/dz
    # vz -- partial dv/dz
    # wz -- partial dw/dz
    # T -- temperature
    # Tz -- partial dT/dz
    # p -- pressure
    problem = de.IVP(domain, variables=['u', 'v', 'w', 'uz', 'vz', 'wz', 'T', 'Tz', 'p', 'w_avgt'])
    problem.parameters['Ra'] = params["Ra"]
    problem.parameters['Pr'] = params["Pr"]
    problem.parameters['Ek'] = params["Ek"]
    problem.parameters['Theta'] = params["Theta"]
    problem.parameters['Lx'] = params["Lx"]
    problem.parameters['Lz'] = params["Lz"]
    problem.parameters['R'] = params["R"]

    # Nondimensionalised Boussinesq equations
    problem.add_equation("dt(u) + dx(p) - dx(dx(u)) - dz(uz) - 2*v*sin(Theta)/Ek = -u*dx(u) - w*uz")
    problem.add_equation("dt(v) - dx(dx(v)) - dz(vz) - 2*w*cos(Theta)/Ek + 2*u*sin(Theta)/Ek = -u*dx(v) - w*vz")
    problem.add_equation("dt(w) + dz(p) - dx(dx(w)) - dz(wz) + v*cos(Theta)/Ek - Ra/Pr * T = -u*dx(w) - w*wz")

    # Convection-diffusion equation, governs evolution of temperature field
    problem.add_equation("dt(T) - (dx(dx(T)) + dz(Tz)) / Pr = -u*dx(T) - w*Tz")

    # Continuity equation
    problem.add_equation("dx(u) + wz = 0")

    # Substitutions for derivatives
    problem.add_equation("dz(u) - uz = 0")
    problem.add_equation("dz(v) - vz = 0")
    problem.add_equation("dz(w) - wz = 0")
    problem.add_equation("dz(T) - Tz = 0")

    # Boundary conditions
    problem.add_bc("left(Tz) = -4")
    problem.add_bc("right(T) = 0")
    problem.add_bc("left(u) = 0")
    problem.add_bc("left(v) = 0")
    problem.add_bc("left(w) = 0")
    problem.add_bc("right(u) = 0")
    problem.add_bc("right(v) = 0")
    problem.add_bc("right(w) = 0", condition="(nx != 0)")
    problem.add_bc("right(p) = 0", condition="(nx == 0)")
    
    problem.add_equation("dt(w_avgt) = w")

    solver = problem.build_solver("RK222")
    
    
    ##################################################
    # Initialise fields, either from previous sim or afresh
    
    if path.exists(path.join(data_dir, 'analysis.h5')):
        print("Analysis file already exists in the provided directory, will continue from where this simulation ended")
        filepath = path.join(data_dir, 'analysis.h5')
        solver.load_state(filepath)
        shutil.move(filepath, path.join(data_dir, 'analysis_previous.h5'))
    else:
        # We need to create a perturbation in the initial temperature field
        wavelength = params["Lx"]
        initial_T = lambda x, z: 0.1 + 0.1 * np.sin(2 * np.pi * x / (wavelength * params["resX"] / params["Lx"]))
        T = solver.state['T']
        Tz = solver.state['Tz']
        for z in range(0, params["resZ"]):
            for x in range(0, params["resX"]):
                T['g'][x][z] = initial_T(x, z)
        T.differentiate('z', out=Tz)
    
    
    ##################################################
    # Prepare directory for simulation results
    
    # shutil.rmtree(data_dir, ignore_errors=True)
    analysis = solver.evaluator.add_file_handler(data_dir, sim_dt=params["timestep_analysis"], max_writes=200, mode='overwrite')
    analysis.add_system(solver.state, layout='g')
    analysis.add_task("integ(integ(0.5 * (u*u + w*w), 'x'), 'z') / (Lx * Lz)", layout='g', name='E')
    analysis.add_task("T * w - Tz", layout='g', name='FluxHeat')
    analysis.add_task("integ(T * w, 'x') / Lx", layout='g', name='FluxHeatConv')
    analysis.add_task("integ(-Tz, 'x') / Lx", layout='g', name='FluxHeatCond')
    
    
    ##################################################
    # Configure CFL to adjust timestep dynamically
    
    CFL = flow_tools.CFL(solver, initial_dt=params["timestep"], cadence=10, safety=0.5, max_change=1.5, min_change=0.5, max_dt=1e-4, threshold=0.05)
    CFL.add_velocities(('u', 'w'))
    flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
    # flow.add_property("sqrt(u*u + w*w)", name='Re')
    # flow.add_property("T", name='T')
    # flow.add_property("-Tz", name='FluxCond')
    # flow.add_property("T * w", name='FluxConv')
    
    
    ##################################################
    # Run the simulation
    
    solver.stop_sim_time = params["duration"]
    solver.stop_wall_time = np.inf
    solver.stop_iteration = np.inf
    print("Simulation start")
    world_start_time = solver.get_world_time()
    t0 = time.time()
    while solver.proceed:
        dt = CFL.compute_dt()
        dt = solver.step(dt)
        if (solver.iteration - 1) % 10 == 0:
            world_run_time = solver.get_world_time() - world_start_time
            ETA = 1/60 * (params["duration"] - solver.sim_time) * world_run_time / solver.sim_time
            print(''.join([' '] * 200), end='\r')
            print(
                'Completed iteration {} (t = {:.3E}, dt = {:.3E}, {:.1f}%) ETA = {:.2f} min'
                .format(solver.iteration, solver.sim_time, dt, 100 * solver.sim_time / params["duration"], ETA),
            end='\r', flush=True)

    print(f'Simulation finished in {time.time() - t0} seconds, merging files...')
    
    
    ##################################################
    # Merge the output files together
    
    post.merge_process_files(data_dir, cleanup=True)
    set_paths = glob(path.join(data_dir, '*.h5'))
    post.merge_sets("{}/analysis.h5".format(data_dir), set_paths, cleanup=True)
    
    print(f'Finished merging files, total time: {time.time() - t0} seconds')


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please provide one argument: The file path to the directory to save the analysis in.")
        exit(1)
    data_dir = sys.argv[1]
    run(data_dir)
