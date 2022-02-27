#!/usr/bin/env python3

import h5py
import sys
from os import path
import numpy as np
from dedalus import public as de
from dedalus.tools import post
from glob import glob
import shutil
import time

import utils
import filtering

import logging
logger = logging.getLogger(__name__)

def main(data_dir):

    # Read parameters from file
    params = utils.read_params(data_dir)

    with h5py.File(path.join(data_dir, 'state.h5'), mode='r') as state_file:

        dims = state_file['tasks']['u'].dims
        num_dims = len(dims) - 1

        # Read in original un-interpolated fields and the dimension scales
        logger.info("Reading fields from hdf5 file...")
        u = np.array(state_file['tasks']['u'])
        v = np.array(state_file['tasks']['v']) if num_dims == 3 else None
        w = np.array(state_file['tasks']['w'])

        t = np.array(dims[0]["sim_time"])
        x = np.array(dims[1][0])
        y = np.array(dims[2][0]) if num_dims == 3 else None
        z = np.array(dims[3 if num_dims == 3 else 2][0])

        # Swap the x and z axes of the velocities we read from the file, so they match our domain
        u = np.swapaxes(u, 1, -1)
        v = np.swapaxes(v, 1, -1)
        w = np.swapaxes(w, 1, -1)

        # Set up an all-Fourier domain for the output
        logger.info("Setting up outputs...")
        xbasis = de.Fourier('x', len(x), interval=(x[0], x[-1]))
        ybasis = de.Fourier('y', len(y), interval=(y[0], y[-1])) if y is not None else None
        zbasis = de.Fourier('z', len(z), interval=(z[0], z[-1]))
        # We place the z axis (the one we are interpolating along) in the first dimension, as
        # this is the one that is non-distributed (local) in Dedalus, i.e. it is not split across
        # processes. This makes it possible to interpolate along the entire axis with a Dedalus operator
        domain = de.Domain((zbasis, ybasis, xbasis) if ybasis is not None else (xbasis, zbasis), grid_dtype=np.float64, mesh=params["mesh"])

        # 'Crop' the read fields to the sub-domain that this process is handling
        distcoords = np.array([0, *domain.dist.coords])
        l_shape = np.array(domain.local_grid_shape())
        l_offsets = distcoords * l_shape
        u = u[:, l_offsets[0] : l_offsets[0] + l_shape[0], l_offsets[1] : l_offsets[1] + l_shape[1], l_offsets[2] : l_offsets[2] + l_shape[2]]
        v = v[:, l_offsets[0] : l_offsets[0] + l_shape[0], l_offsets[1] : l_offsets[1] + l_shape[1], l_offsets[2] : l_offsets[2] + l_shape[2]]
        w = w[:, l_offsets[0] : l_offsets[0] + l_shape[0], l_offsets[1] : l_offsets[1] + l_shape[1], l_offsets[2] : l_offsets[2] + l_shape[2]]

        # Set up a (Chebyshev, Fourier, Fourier) domain for the interpolation input
        zbasis_cheby = de.Chebyshev('z', len(z), interval=(z[0], z[-1]))
        domain_cheby = de.Domain((zbasis_cheby, ybasis, xbasis), mesh=params["mesh"])

        # Create the GeneralFunction operator for interpolating the z axis
        def interp_to_fourier(field):
            original_field_layout = field.layout
            field_cheby = domain_cheby.new_field(name=field.name)
            field_cheby['g'] = field['g']
            interped = filtering.interp_to_basis_dedalus(field_cheby, dest=de.Fourier, axis=0)
            # Put the field back in the layout it was originally, Dedalus can complain if we don't
            field.require_layout(original_field_layout)
            return interped
        def interp_to_fourier_wrapper(field):
            return de.operators.GeneralFunction(
                field.domain,
                layout='g',
                func=interp_to_fourier,
                args=(field,)
            )
        de.operators.parseables['interp_to_fourier'] = interp_to_fourier_wrapper

        # Set up an essentially 'empty' problem
        problem = de.IVP(domain,variables=['u', 'v', 'w'])
        problem.add_equation("dt(u) = 0")
        problem.add_equation("dt(v) = 0")
        problem.add_equation("dt(w) = 0")
        solver = problem.build_solver(de.timesteppers.RK443)

        # Create the file handler to output the interpolated fields
        handler = solver.evaluator.add_file_handler(path.join(data_dir, "interp"), mode='overwrite')
        handler.last_wall_div = handler.last_sim_div = handler.last_iter_div = 0
        handler.add_task("interp_to_fourier(u)", layout='g', name='u')
        handler.add_task("interp_to_fourier(v)", layout='g', name='v')
        handler.add_task("interp_to_fourier(w)", layout='g', name='w')

        time_start = time.time()

        # Start evolving the system
        last_t = 0
        log_interval = 60
        last_log = time_start - log_interval - 1
        logger.info("Running interpolation, logging every {}s".format(log_interval))
        for i in range(len(t)):
            current_t = t[i]
            dt = current_t - last_t
            solver.step(dt)
            last_t = t[i]
            solver.state['u']['g'] = u[i]
            if v is not None: solver.state['v']['g'] = v[i]
            solver.state['w']['g'] = w[i]
            solver.evaluator.evaluate_handlers((handler,), world_time=0, wall_time=0, sim_time=solver.sim_time, timestep=dt, iteration=i)
            time_now = time.time()
            if time_now - last_log > log_interval:
                last_log = time_now
                logger.info("{}%".format(np.round(100 * i / len(t), 0)))

        time_end = time.time()
        logger.info("Finished interpolating in {} minutes".format(np.round((time_end - time_start) / 60, 2)))

        # Merge the files
        logger.info("Merging output files...")
        interp_dir = path.join(data_dir, "interp")
        post.merge_process_files(interp_dir, cleanup=True)
        set_paths = glob(path.join(interp_dir, '*.h5'))
        post.merge_sets(path.join(data_dir, 'interp.h5'), set_paths, cleanup=True)

        logger.info("Done")

if __name__ == '__main__':

    if len(sys.argv) < 2:
        logger.info("Please provide one argument: The file path to the directory to read the analysis from.")
        exit(1)
    data_dir = sys.argv[1]

    main(data_dir)
    # test(data_dir)
