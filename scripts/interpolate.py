#!/usr/bin/env python3

import h5py
import sys
from os import path
import numpy as np
import time
from mpi4py import MPI

import utils
import filtering

import logging
logger = logging.getLogger(__name__)


def interp_along_axis(arr, axis, basis):
    arr = np.swapaxes(arr, 0, axis)

    interped = np.zeros(arr.shape)
    res = len(basis)
    basis_lin = np.linspace(basis[0], basis[-1], res)

    # for a index i in the linear basis, we will interpolate between
    # i_src and i_src+1 in the original basis
    i_src = 0
    for i in range(res):
        x = basis_lin[i]
        # find the points in the original data that we want to interpolate between
        while i_src < res - 2 and basis[i_src + 1] < x:
            i_src += 1
        y0 = arr[i_src]
        y1 = arr[i_src + 1]
        x0 = basis[i_src]
        x1 = basis[i_src + 1]
        gradient = (y1 - y0) / (x1 - x0)
        interped[i] = y0 + (x - x0) * gradient

    interped = np.swapaxes(interped, 0, axis)
    return interped


def interp(data_dir, f):

    # Read parameters from file
    params = utils.read_params(data_dir)

    comm = MPI.COMM_WORLD
    num_ranks = comm.size
    rank = comm.Get_rank()

    senddata = None
    timesteps = None
    sizes = None
    offsets = None
    x = np.empty(params['resX'], dtype='d')
    y = np.empty(params['resY'], dtype='d')
    z = np.empty(params['resZ'], dtype='d')
    if rank == 0:
        with h5py.File(path.join(data_dir, 'state.h5'), mode='r') as state_file:
            dims = state_file['tasks'][f].dims
            # Read in original un-interpolated fields and the dimension scales
            logger.info("Reading field {} from hdf5 file...".format(f))
            vel = np.array(state_file['tasks'][f], dtype='d')
            t = np.array(dims[0]["sim_time"])
            x = np.array(dims[1][0], dtype='d')
            y = np.array(dims[2][0], dtype='d')
            z = np.array(dims[3][0], dtype='d')
            logger.info("Total size: {} timesteps".format(vel.shape[0]))
            duration = min(params['duration'], t[-1])
            if duration < params['average_interval']:
                print('WARNING: averaging interval longer than simulation duration, averaging over entire duration...')
            timeframe_mask = np.logical_and(t >= duration - params['average_interval'], t <= duration)
            t = t[timeframe_mask]
            vel = vel[timeframe_mask]
            logger.info("Cropped to {} timesteps, according to 'average_interval' parameter".format(len(t)))

            if num_ranks > len(t):
                logger.info("Require num processes <= num timesteps")
                exit(1)

        # Split up the timesteps equally across all ranks, and give the remaining ones to rank 0
        # senddata = np.array(np.array_split(vel, num_ranks))
        senddata = vel
        timesteps = [vel.shape[0] // num_ranks] * (num_ranks - 1)
        timesteps.append(vel.shape[0] - timesteps[0] * (num_ranks - 1))
        sizes = [len(x) * len(y) * len(z) * len_t for len_t in timesteps]
        offsets = np.insert(np.cumsum(sizes), 0, 0)[0:-1]

    logging.info("Scattering data...")
    comm.Bcast(x, root=0)
    comm.Bcast(y, root=0)
    comm.Bcast(z, root=0)
    num_timesteps = comm.scatter(timesteps, root=0)
    vel = np.empty((num_timesteps, params['resX'], params['resY'], params['resZ']), dtype='d')
    comm.Scatterv([senddata, sizes, offsets, MPI.DOUBLE], vel, root=0)

    print("Rank {} has array of shape {}, starting interpolation and filtering".format(rank, vel.shape))

    # wavelength = params['Lz'] / 2
    wavelength = 0.113137
    lowpass = np.zeros(vel.shape)
    highpass = np.zeros(vel.shape)
    z_fourier = np.linspace(z[0], z[-1], len(z))
    for i in range(vel.shape[0]):
        vel[i] = interp_along_axis(vel[i], -1, z)
        lowpass[i] = filtering.kspace_lowpass(vel[i], (0, 1, 2), (x, y, z_fourier), wavelength)
        highpass[i] = filtering.kspace_highpass(vel[i], (0, 1, 2), (x, y, z_fourier), wavelength)
        # if i % (vel.shape[0] // 5) == 0:
            # print("Rank {} is at {}%".format(rank, np.round(100 * i / vel.shape[0], 1)))

    comm.Barrier()
    logger.info("Finished interpolation, gathering on rank 0...")

    vel_all = lowpass_all = highpass_all = None
    if rank == 0:
        vel_all = np.empty((len(t), len(x), len(y), len(z)), dtype='d')
        lowpass_all = np.empty((len(t), len(x), len(y), len(z)), dtype='d')
        highpass_all = np.empty((len(t), len(x), len(y), len(z)), dtype='d')
    comm.Gatherv(vel, [vel_all, sizes, offsets, MPI.DOUBLE], root=0)
    comm.Gatherv(lowpass, [lowpass_all, sizes, offsets, MPI.DOUBLE], root=0)
    comm.Gatherv(highpass, [highpass_all, sizes, offsets, MPI.DOUBLE], root=0)

    if rank == 0:
        logger.info("Rank 0 has complete velocity field of shape {}".format(vel_all.shape))
        filepath = path.join(data_dir, "interp_{}.h5".format(f))
        logger.info("Saving to {}".format(filepath))
        # Now have the complete interpolation, output to h5 file
        with h5py.File(filepath, 'w') as interp_file:
            scales = interp_file.create_group("scales")
            scale_t = scales.create_dataset('t', data=t).make_scale()
            scale_x = scales.create_dataset('x', data=x).make_scale()
            scale_y = scales.create_dataset('y', data=y).make_scale()
            scale_z = scales.create_dataset('z', data=z_fourier).make_scale()
            tasks = interp_file.create_group("tasks")
            datasets = (
                tasks.create_dataset(f, data=vel_all),
                tasks.create_dataset("{}_lowpass".format(f), data=lowpass_all),
                tasks.create_dataset("{}_highpass".format(f), data=highpass_all)
            )
            for dataset in datasets:
                dataset.dims[0].attach_scale(interp_file['scales']['t'])
                dataset.dims[1].attach_scale(interp_file['scales']['x'])
                dataset.dims[2].attach_scale(interp_file['scales']['y'])
                dataset.dims[3].attach_scale(interp_file['scales']['z'])

    comm.Barrier()
    logger.info("Done")

if __name__ == '__main__':

    if len(sys.argv) < 2:
        logger.info("Please provide one argument: The file path to the directory to read the analysis from.")
        exit(1)
    data_dir = sys.argv[1]

    fields = ['u', 'v', 'w']
    for f in fields:
        interp(data_dir, f)

