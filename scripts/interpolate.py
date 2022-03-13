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
from mpi4py import MPI

import utils
import filtering

import logging
logger = logging.getLogger(__name__)

def interp(data_dir, f):

    # Read parameters from file
    params = utils.read_params(data_dir)

    # Set up an all-Fourier domain for the output
    logger.info("Setting up outputs...")
    xbasis = de.Fourier('x',params["resX"],interval=(-params["Lx"]/2.,params["Lx"]/2.))
    ybasis = de.Fourier('y',params["resY"],interval=(-params["Ly"]/2.,params["Ly"]/2.))
    zbasis = de.Fourier('z',params["resZ"],interval=(-params["Lz"]/2.,params["Lz"]/2.))
    # We place the z axis (the one we are interpolating along) in the first dimension, as
    # this is the one that is non-distributed (local) in Dedalus, i.e. it is not split across
    # processes. This makes it possible to interpolate along the entire axis with a Dedalus operator
    domain = de.Domain((zbasis, ybasis, xbasis), grid_dtype=np.float64, mesh=params["mesh"])

    distcoords = np.array([0, *domain.dist.coords]) # The part of the mesh that this process is tackling
    l_shape = np.array(domain.local_grid_shape()) # The shape of the local grid for this process
    l_offsets = distcoords * l_shape # The position of the local grid within the global one for this process

    comm = MPI.COMM_WORLD
    num_ranks = comm.size
    rank = comm.Get_rank()
    if rank == 0:
        with h5py.File(path.join(data_dir, 'state.h5'), mode='r') as state_file:

            dims = state_file['tasks'][f].dims
            num_dims = len(dims) - 1

            # Read in original un-interpolated fields and the dimension scales
            print("Rank 0 is reading field {} from hdf5 file...".format(f))
            vel = np.array(state_file['tasks'][f])

            t = np.array(dims[0]["sim_time"])
            x = np.array(dims[1][0])
            y = np.array(dims[2][0]) if num_dims == 3 else None
            z = np.array(dims[3 if num_dims == 3 else 2][0])

            # Swap the x and z axes of the velocities we read from the file, so they match our domain
            vel = np.swapaxes(vel, 1, -1)

            for r in range(1, num_ranks):
                # 'Crop' the read fields to the sub-domain that this rank should handle
                data = comm.recv(source=r)
                shape = data['l_shape']
                offsets = data['l_offsets']
                rank_vel = vel[:, offsets[0] : offsets[0] + shape[0], offsets[1] : offsets[1] + shape[1], offsets[2] : offsets[2] + shape[2]]
                # Send the local grid data to the rank that asked for it
                comm.send({
                    'vel': rank_vel,
                    'dims': [t, x, y, z]
                }, dest=r)

            # Get the local grid data for rank 0, so it can participate too
            vel = vel[:, l_offsets[0] : l_offsets[0] + l_shape[0], l_offsets[1] : l_offsets[1] + l_shape[1], l_offsets[2] : l_offsets[2] + l_shape[2]]
    else:
        comm.send({
            'l_shape': l_shape,
            'l_offsets': l_offsets
        }, dest=0)
        data = comm.recv(source=0)
        vel = data['vel']
        t, x, y, z = data['dims']

    print("Rank {} has grid data of shape {}, t {}, x {}, y {}, z {}".format(rank, vel.shape, t.shape, x.shape, y.shape, z.shape))

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
    problem = de.IVP(domain,variables=[f])
    problem.add_equation("dt({}) = 0".format(f))
    solver = problem.build_solver(de.timesteppers.RK443)

    # Create the file handler to output the interpolated fields
    handler = solver.evaluator.add_file_handler(path.join(data_dir, "interp_{}".format(f)), mode='overwrite')
    handler.last_wall_div = handler.last_sim_div = handler.last_iter_div = 0
    handler.add_task("interp_to_fourier({})".format(f), layout='g', name=f)

    time_start = time.time()

    # Start evolving the system
    last_t = 0
    log_interval = 30
    last_log = time_start - log_interval - 1
    print("rank {}: Running interpolation, logging every {}s".format(rank, log_interval))
    for i in range(len(t)):
        current_t = t[i]
        dt = current_t - last_t
        solver.step(dt)
        last_t = t[i]
        solver.state[f]['g'] = vel[i]
        solver.evaluator.evaluate_handlers((handler,), world_time=0, wall_time=0, sim_time=solver.sim_time, timestep=dt, iteration=i)
        time_now = time.time()
        if time_now - last_log > log_interval:
            last_log = time_now
            print("rank{}: {}%".format(rank, np.round(100 * i / len(t), 0)))

    time_end = time.time()
    print("rank {}: Finished interpolating in {} minutes".format(rank, np.round((time_end - time_start) / 60, 2)))

    # Merge the files
    print("rank {}: Merging output files...".format(rank))
    interp_dir = path.join(data_dir, "interp_{}".format(f))
    post.merge_process_files(interp_dir, cleanup=True)
    set_paths = glob(path.join(interp_dir, '*.h5'))
    post.merge_sets(path.join(data_dir, 'interp_{}.h5'.format(f)), set_paths, cleanup=True)

    logger.info("rank {}: Done".format(rank))

if __name__ == '__main__':

    if len(sys.argv) < 2:
        logger.info("Please provide one argument: The file path to the directory to read the analysis from.")
        exit(1)
    data_dir = sys.argv[1]

    fields = ['u', 'v', 'w']
    for f in fields:
        interp(data_dir, f)
