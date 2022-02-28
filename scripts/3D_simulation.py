#!/usr/bin/env python3

import numpy as np
from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.tools import post
import shutil
from glob import glob
import os.path as path
from mpi4py import MPI
import time
import sys

import utils
import filtering

import logging
logger = logging.getLogger(__name__)

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
    problem.add_equation("dt(u) + dx(p) - (dx(dx(u)) + dy(dy(u)) + dz(uz)) + (Ta ** 0.5) * v * cos(Theta) = - (u * dx(u) + v * dy(u) + w * uz)")
    # y-component of the momentum equation
    problem.add_equation("dt(v) + dy(p) - (dx(dx(v)) + dy(dy(v)) + dz(vz)) - (Ta ** 0.5) * (u * cos(Theta) + w * sin(Theta)) = - (u * dx(v) + v * dy(v) + w * vz)")
    # z-component of the momentum equation
    problem.add_equation("dt(w) + dz(p) - (dx(dx(w)) + dy(dy(w)) + dz(wz)) + (Ta ** 0.5) * v * sin(Theta) - X * T = -(u * dx(w) + v * dy(w) + w * wz)")
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
    logger.info('Solver built')
    
    ##################################################
    # Initialise fields, either from previous sim or afresh
    
    if path.exists(path.join(data_dir, 'state.h5')):
        filepath = path.join(data_dir, 'state.h5')
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

    state = solver.evaluator.add_file_handler(path.join(data_dir, "state"), sim_dt=params["timestep_analysis"], mode='overwrite')
    state.add_system(solver.state, layout='g')
    state.add_task("ut", layout='g', name='u_dt')
    state.add_task("vt", layout='g', name='v_dt')

    analysis = solver.evaluator.add_file_handler(path.join(data_dir, "analysis"), sim_dt=params["timestep_analysis"], mode='overwrite')
    # Total energy E(t)
    analysis.add_task("integ(integ(integ(0.5 * (u*u + v*v + w*w), 'x'), 'y'), 'z') / (Lx * Ly * Lz)", layout='g', name='E')
    # Vertical heat fluxes F(z, t)
    analysis.add_task("integ(integ(T * w, 'x'), 'y') / (Lx * Ly)", layout='g', name='FluxHeatConv')
    analysis.add_task("integ(integ(-Tz, 'x'), 'y') / (Lx * Ly)", layout='g', name='FluxHeatCond')
    # Horizontally averaged velocities U(z, t)
    # Also the coriolis terms in the averaged momentum equations
    # coriolis_x = -v, coriolis_y = u
    analysis.add_task("integ(integ(u, 'x'), 'y') / (Lx * Ly)", layout='g', name='MeanU')
    analysis.add_task("integ(integ(v, 'x'), 'y') / (Lx * Ly)", layout='g', name='MeanV')
    analysis.add_task("integ(integ(w, 'x'), 'y') / (Lx * Ly)", layout='g', name='MeanW')
    # Terms of averaged momentum equation q(z, t)
    analysis.add_task("-integ(integ(dz(uz) / (Ta**0.5 * sin(Theta)), 'x'), 'y') / (Lx * Ly)", layout='g', name='ViscousX')
    analysis.add_task("-integ(integ(dz(vz) / (Ta**0.5 * sin(Theta)), 'x'), 'y') / (Lx * Ly)", layout='g', name='ViscousY')
    analysis.add_task("integ(integ(ut / (Ta**0.5 * sin(Theta)), 'x'), 'y') / (Lx * Ly)", layout='g', name='TemporalX')
    analysis.add_task("integ(integ(vt / (Ta**0.5 * sin(Theta)), 'x'), 'y') / (Lx * Ly)", layout='g', name='TemporalY')
    analysis.add_task("integ(integ(v + (dz(uz) - ut) / (Ta**0.5 * sin(Theta)), 'x'), 'y') / (Lx * Ly)", layout='g', name='StressX')
    analysis.add_task("integ(integ(-u + (dz(vz) - vt) / (Ta**0.5 * sin(Theta)), 'x'), 'y') / (Lx * Ly)", layout='g', name='StressY')
    
    ##################################################
    # Configure CFL to adjust timestep dynamically
    
    CFL = flow_tools.CFL(solver, initial_dt=params["timestep"], cadence=10, safety=0.5, max_change=1.5, min_change=0.5, max_dt=1e-4, threshold=0.05)
    CFL.add_velocities(('u', 'v', 'w'))
    flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
    flow.add_property("sqrt(u**2 + v**2 + w**2)/Ra", name='Re')
    
    ##################################################
    # Run the simulation
    
    solver.stop_sim_time = params["duration"]
    solver.stop_wall_time = np.inf
    solver.stop_iteration = np.inf
    try:
        logger.info('Starting loop')
        start_time = time.time()

        while solver.ok:
            dt = CFL.compute_dt()
            dt = solver.step(dt)

            if (solver.iteration) == 1:
                # Prints various parameters to terminal upon starting the simulation
                logger.info('Parameter values imported from params.json:')
                logger.info('Lx = {}, Ly = {}, Lz = {}; (Resolution of {},{},{})'.format(params["Lx"], params["Ly"], params["Lz"], params["resX"], params["resY"], params["resZ"]))
                logger.info('Ra = {}, Pr = {}, Ta = {}, Theta = {}'.format(params["Ra"], params["Pr"], params["Ta"], params["Theta"]))
                logger.info('Files outputted every {}'.format(params["timestep_analysis"]))
                if params["duration"] != np.inf:
                    logger.info('Simulation finishes at sim_time = {}'.format(params["duration"]))
                else:
                    logger.info('No clear end point defined. Simulation may run perpetually.')

            if (solver.iteration-1) % 100 == 0:
                # Prints progress information include maximum Reynolds number every 100 iterations
                logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
                logger.info('Max Re = %f' %flow.max('Re'))  

    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        # Prints concluding information upon reaching the end of the simulation.
        end_time = time.time()
        logger.info('Iterations: %i' %solver.iteration)
        logger.info('Sim end time: %f' %solver.sim_time)
        logger.info('Run time: %.2f sec' %(end_time-start_time))
        logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))

        
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please provide one argument: The file path to the directory to save the analysis in.")
        exit(1)
    data_dir = sys.argv[1]
    run(data_dir)
    merge(data_dir)
