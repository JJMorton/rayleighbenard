#!/usr/bin/env python3

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

    def create_filter(axis, cutoff_freq, filtertype):
        def filter_func(field):
            interval = field.domain.bases[axis].interval
            res = field.domain.bases[axis].base_grid_size
            sample_spacing = (interval[1] - interval[0]) / res
            nyq = 0.5 / sample_spacing
            cutoff = cutoff_freq / nyq
            # b, a = signal.butter(8, cutoff, btype='low', analog=False, output='ba')
            sos = signal.butter(8, cutoff, btype=filtertype, analog=False, output='sos')
            # data = signal.filtfilt(b, a, data, axis=axis)
            filtered = signal.sosfiltfilt(sos, field.data, axis=axis)
            return filtered
        return filter_func
    
    def lowpass_x(field):
        # Filters out wavelengths shorter (frequencies higher) than this cutoff
        cutoff_wavelength = 0.7 * params['Lx']
        return de.operators.GeneralFunction(
            field.domain,
            layout='g',
            func=create_filter(0, 1/cutoff_wavelength, 'low'),
            args=(field,)
        )
        
    de.operators.parseables['lowpass_x'] = lowpass_x

    def highpass_x(field):
        # Filters out wavelengths longer (frequencies lower) than this cutoff
        cutoff_wavelength = 0.7 * params['Lx']
        return de.operators.GeneralFunction(
            field.domain,
            layout='g',
            func=create_filter(0, 1/cutoff_wavelength, 'high'),
            args=(field,)
        )
            
    de.operators.parseables['highpass_x'] = highpass_x
    
    
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
    # average_fields = ['u_avgt', 'v_avgt', 'w_avgt', 'u_pert', 'v_pert', 'w_pert', 'stress_uw', 'stress_uw_low', 'stress_uw_high', 'stress_vw']
    problem = de.IVP(domain, variables=['u', 'v', 'w', 'uz', 'vz', 'wz', 'T', 'Tz', 'p', 'ut', 'vt'])#, *average_fields])
    problem.parameters['Ra'] = params["Ra"]
    problem.parameters['Pr'] = params["Pr"]
    problem.parameters['Ta'] = params["Ta"]
    problem.parameters['Theta'] = params["Theta"]
    problem.parameters['Lx'] = params["Lx"]
    problem.parameters['Lz'] = params["Lz"]
    # problem.parameters['Tau'] = params["average_interval"] # The time period over which we want to average

    # Nondimensionalised Boussinesq equations
    problem.add_equation("dt(u) + dx(p) - dx(dx(u)) - dz(uz) - v*sin(Theta) * Ta**0.5                               = -u*dx(u) - w*uz")
    problem.add_equation("dt(v)         - dx(dx(v)) - dz(vz) + (u*sin(Theta) - w*cos(Theta)) * Ta**0.5              = -u*dx(v) - w*vz")
    problem.add_equation("dt(w) + dz(p) - dx(dx(w)) - dz(wz) - v*cos(Theta) * Ta**0.5                   - Ra/Pr * T = -u*dx(w) - w*wz")

    # Convection-diffusion equation, governs evolution of temperature field
    problem.add_equation("dt(T) - (dx(dx(T)) + dz(Tz)) / Pr = -u*dx(T) - w*Tz")

    # Continuity equation
    problem.add_equation("dx(u) + wz = 0")

    # Substitutions for derivatives
    problem.add_equation("dz(u) - uz = 0")
    problem.add_equation("dz(v) - vz = 0")
    problem.add_equation("dz(w) - wz = 0")
    problem.add_equation("dz(T) - Tz = 0")
    problem.add_equation("dt(u) - ut = 0")
    problem.add_equation("dt(v) - vt = 0")

    # Boundary conditions
    problem.add_bc("left(Tz) = -1")
    # problem.add_bc("left(T) = 20")
    problem.add_bc("right(T) = 0")
    problem.add_bc("left(uz) = 0")
    problem.add_bc("left(vz) = 0")
    problem.add_bc("left(w) = 0")
    problem.add_bc("right(uz) = 0")
    problem.add_bc("right(vz) = 0")
    problem.add_bc("right(w) = 0", condition="(nx != 0)")
    problem.add_bc("right(p) = 0", condition="(nx == 0)")

    # # Time averaged quantities
    # problem.add_equation("dt(w_avgt) = w / Tau")
    # problem.add_equation("dt(u_avgt) = u / Tau")
    # problem.add_equation("dt(v_avgt) = v / Tau")
    # problem.add_equation("u_pert = u - u_avgt")
    # problem.add_equation("v_pert = v - v_avgt")
    # problem.add_equation("w_pert = w - w_avgt")
    # problem.add_equation("dt(stress_uw) = w_pert*u_pert / Tau")
    # problem.add_equation("dt(stress_uw_low) = lowpass_x(w_pert)*lowpass_x(u_pert) / Tau")
    # problem.add_equation("dt(stress_uw_high) = highpass_x(w_pert)*highpass_x(u_pert) / Tau")
    # problem.add_equation("dt(stress_vw) = w_pert*v_pert / Tau")

    solver = problem.build_solver("RK222")
    
    
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
    analysis.add_task("integ(integ(0.5 * (u*u + w*w), 'x'), 'z') / (Lx * Lz)", layout='g', name='E')
    analysis.add_task("T * w - Tz", layout='g', name='FluxHeat')
    analysis.add_task("integ(T * w, 'x') / Lx", layout='g', name='FluxHeatConv')
    analysis.add_task("integ(-Tz, 'x') / Lx", layout='g', name='FluxHeatCond')
    # Derivatives seem to be more accurate when calculated in dedalus, rather than in post
    analysis.add_task("dz(u)", layout='g', name='u_dz')
    analysis.add_task("dz(v)", layout='g', name='v_dz')
    analysis.add_task("dz(w)", layout='g', name='w_dz')
    analysis.add_task("dx(u)", layout='g', name='u_dx')
    analysis.add_task("dx(v)", layout='g', name='v_dx')
    analysis.add_task("dx(w)", layout='g', name='w_dx')
    analysis.add_task("dz(dz(u))", layout='g', name='u_dz2')
    analysis.add_task("dz(dz(v))", layout='g', name='v_dz2')
    analysis.add_task("dz(dz(w))", layout='g', name='w_dz2')
    analysis.add_task("ut", layout='g', name='u_dt')
    analysis.add_task("vt", layout='g', name='v_dt')
    
    # # Set interval of infinity so that that handler is not called automatically
    # # We will manually call the handler, so that we can reset the averages to zero straight after
    # averaged = solver.evaluator.add_file_handler(path.join(data_dir, 'averaged'), sim_dt=np.inf, mode='overwrite')
    # # Warning: dirty hack
    # # Make dedalus think this handler has been evaluated at time t=0
    # # This stops the handlers being evaluated at the start:
    # averaged.last_wall_div = averaged.last_sim_div = averaged.last_iter_div = 0
    # averaged.add_task("u_avgt", layout='g', name='u')
    # averaged.add_task("v_avgt", layout='g', name='v')
    # averaged.add_task("w_avgt", layout='g', name='w')
    # averaged.add_task("stress_uw", layout='g', name='stress_uw')
    # averaged.add_task("stress_vw", layout='g', name='stress_vw')
    # averaged.add_task("dz(integ(stress_uw, 'x') / Lx)", layout='g', name='stress_uw_avgx_dz')
    # averaged.add_task("dz(integ(stress_vw, 'x') / Lx)", layout='g', name='stress_vw_avgx_dz')
    # averaged.add_task("stress_uw_low", layout='g', name='stress_uw_low')
    # averaged.add_task("stress_uw_high", layout='g', name='stress_uw_high')
    # averaged.add_task("dz(integ(stress_uw_low, 'x') / Lx)", layout='g', name='stress_uw_low_avgx_dz')
    # averaged.add_task("dz(integ(stress_uw_high, 'x') / Lx)", layout='g', name='stress_uw_high_avgx_dz')
    
    
    ##################################################
    # Configure CFL to adjust timestep dynamically
    
    CFL = flow_tools.CFL(solver, initial_dt=params["timestep"], cadence=10, safety=0.5, max_change=1.5, min_change=0.5, max_dt=1e-4, threshold=0.05)
    CFL.add_velocities(('u', 'w'))
    flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
    
    
    ##################################################
    # Run the simulation
    
    # def reset_averages():
    #     for var in average_fields:
    #         field = solver.state[var]
    #         nx, nz = np.array(field['g']).shape
    #         field['g'] = [[0]*nz]*nx
    #     print(f'Reset time-averaged fields at time t={solver.sim_time}')
    
    solver.stop_sim_time = params["duration"]
    solver.stop_wall_time = np.inf
    solver.stop_iteration = np.inf
    print("Simulation start")
    sim_time_start = solver.sim_time
    t0 = time.time()
    # # Average at end of first whole average_interval
    # next_average = (np.ceil(solver.sim_time / params["average_interval"]) + 1) * params["average_interval"]
    dt = params["timestep"]
    # reset_averages()
    while solver.proceed:
        # Step the simulation forwards
        dt = CFL.compute_dt()
        dt = solver.step(dt)
        
        # # When we reach the end of a time-averaging interval, save to analysis file
        # if solver.sim_time >= next_average:
        #     next_average += params["average_interval"]
        #     solver.evaluate_handlers_now(dt, handlers=[averaged])
        #     print(''.join([' '] * 200), end='\r')
        #     print(f'Calculated time-averaged fields at time t={solver.sim_time}')
            
        # # When we reach the start of a time-averaging interval, zero the averages to restart them
        # if solver.sim_time % params["average_interval"] < dt:
        #     print(''.join([' '] * 200), end='\r')
        #     reset_averages()
        
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
