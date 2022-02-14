#!/usr/bin/env python3

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from os import path
from os import mkdir
import sys

import filtering
import utils


def get_field(file, fieldname):
    """Get a field from the .h5 file, with dimensions (t, x, y, z)"""
    arr = np.array(file['tasks'][fieldname])
    if len(arr.shape) == 3:
        # 2d sim, add dummy y dimension
        arr = arr[:, :, np.newaxis, :]
    return arr

def get_dims(file, fieldname):
    """Get the dimension scales associated with the field"""
    task = file['tasks'][fieldname]
    num_dims = len(task.dims)
    dims = []
    # The first dimension is always time, no way to average over it in dedalus
    dims.append(np.array(task.dims[0]['sim_time']))
    # Get the other 2 or 3 dimensions
    for i in range(1, num_dims):
        dims.append(np.array(task.dims[i][0]))
    # Insert a dummy zonal axis if the simulation was 2d
    if num_dims == 3:
        dims.insert(2, None)
    return tuple(dims)

def average_zonal(arr):
    """Average the field along the zonal direction: (t, x, y, z) --> (t, x, z)"""
    dims = len(arr.shape)
    if dims != 4:
        raise Exception(f"Attempt to zonally average array with {dims} dimensions")
    return np.mean(arr, axis=2)

def average_horizontal(arr):
    """Average the field horizonally: (t, x, y, z) --> (t, z)"""
    dims = len(arr.shape)
    if dims != 4:
        raise Exception(f"Attempt to horizontally average array with {dims} dimensions")
    return np.mean(np.mean(arr, axis=1), axis=1)


def plot_velocities(data_dir, plot_dir):
    """Plot the time averaged velocities"""
    image_name = "velocity.png"
    print(f'Plotting "{image_name}"...')
    params = utils.read_params(data_dir)
    with h5py.File(path.join(data_dir, 'state.h5'), mode='r') as file:

        t, x, y, z = get_dims(file, 'u')

        duration = min(params['duration'], t[-1])
        if duration < params['average_interval']: print('WARNING: averaging interval longer than simulation duration, averaging over entire duration...')
        tstart = duration - params['average_interval']
        timeframe_mask = np.logical_and(t >= tstart, t <= duration)

        t = t[timeframe_mask]

        u = average_zonal(get_field(file, 'u')[timeframe_mask])
        v = average_zonal(get_field(file, 'v')[timeframe_mask])
        w = average_zonal(get_field(file, 'w')[timeframe_mask])

        u_avgt = np.mean(u, axis=0)
        v_avgt = np.mean(v, axis=0)
        w_avgt = np.mean(w, axis=0)

        plots_shape = np.array((2, 2))
        plots_size_each = np.array((8, 4))
        fig = plt.figure(figsize=np.flip(plots_shape) * plots_size_each)
        fig.suptitle(f'Averaged zonally and in time from {np.round(tstart, 2)} to {np.round(duration, 2)} viscous times')

        ax = fig.add_subplot(*plots_shape, 1)
        ax.set_title("Time averaged u")
        pcm = ax.pcolormesh(x, z, u_avgt.T, shading='nearest', cmap="CMRmap", label='<u>')
        fig.colorbar(pcm, ax=ax)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_aspect(1)

        ax = fig.add_subplot(*plots_shape, 2)
        ax.set_title("Time averaged v")
        pcm = ax.pcolormesh(x, z, v_avgt.T, shading='nearest', cmap="CMRmap", label='<v>')
        fig.colorbar(pcm, ax=ax)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_aspect(1)

        ax = fig.add_subplot(*plots_shape, 3)
        ax.set_title("Time averaged w")
        pcm = ax.pcolormesh(x, z, w_avgt.T, shading='nearest', cmap="CMRmap", label='<w>')
        fig.colorbar(pcm, ax=ax)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_aspect(1)

        plt.tight_layout()
        plt.savefig(path.join(plot_dir, image_name))
        plt.close()


def plot_temperature(data_dir, plot_dir):
    image_name = "temperature.png"
    print(f'Plotting "{image_name}"...')
    params = utils.read_params(data_dir)
    with h5py.File(path.join(data_dir, 'state.h5'), mode='r') as file:

        t, x, y, z = get_dims(file, 'T')
        is3D = y is not None

        duration = min(params['duration'], t[-1])
        if duration < params['average_interval']: print('WARNING: averaging interval longer than simulation duration, averaging over entire duration...')
        tstart = duration - params['average_interval']
        timeframe_mask = np.logical_and(t >= tstart, t <= duration)

        t = t[timeframe_mask]

        T = get_field(file, 'T')[timeframe_mask]
        T_avg = np.mean(average_zonal(T), axis=0)

        plots_shape = np.array((2, 1))
        plots_size_each = np.array((8, 4))
        fig = plt.figure(figsize=np.flip(plots_shape) * plots_size_each)
        fig.suptitle(f'Cross-section at y=Ly/2, averaged from {np.round(tstart, 2)} to {np.round(duration, 2)} viscous times')

        ax = fig.add_subplot(*plots_shape, 1)
        ax.set_title("Time & zonally averaged T")
        pcm = ax.pcolormesh(x, z, T_avg.T, shading='nearest', cmap="CMRmap", label='<T>')
        fig.colorbar(pcm, ax=ax)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_aspect(1)

        ax = fig.add_subplot(*plots_shape, 2)
        if is3D:
            ax.set_title(f'Snapshot of T at y=Ly/2 and t={np.round(duration)} viscous times')
            T_snapshot = T[-1, :, params["resY"] // 2, :]
        else:
            ax.set_title(f'Snapshot of T at t={np.round(duration, 2)} viscous times')
            T_snapshot = T[-1, :, 0, :]
        pcm = ax.pcolormesh(x, z, T_snapshot.T, shading='nearest', cmap="CMRmap", label='T')
        fig.colorbar(pcm, ax=ax)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_aspect(1)

        plt.tight_layout()
        plt.savefig(path.join(plot_dir, image_name))
        plt.close()


def plot_heat_flux_z(data_dir, plot_dir):
    image_name = "heat_flux_z.png"
    print(f'Plotting "{image_name}"...')
    params = utils.read_params(data_dir)
    with h5py.File(path.join(data_dir, 'analysis.h5'), mode='r') as file:

        t, _, _, z = get_dims(file, 'FluxHeatConv')

        duration = min(params['duration'], t[-1])
        if duration < params['average_interval']: print('WARNING: averaging interval longer than simulation duration, averaging over entire duration...')
        tstart = duration - params['average_interval']
        timeframe_mask = np.logical_and(t >= tstart, t <= duration)

        t = t[timeframe_mask]

        fluxconv = np.mean(average_horizontal(get_field(file, 'FluxHeatConv')[timeframe_mask]), axis=0)
        fluxcond = np.mean(average_horizontal(get_field(file, 'FluxHeatCond')[timeframe_mask]), axis=0)
        fluxtotal = fluxconv + fluxcond

        plots_shape = np.array((1, 1))
        plots_size_each = np.array((8, 4))
        fig = plt.figure(figsize=np.flip(plots_shape) * plots_size_each)
        fig.suptitle(f'Averaged from {np.round(tstart, 2)} to {np.round(duration, 2)} viscous times')

        ax = fig.add_subplot(*plots_shape, 1)
        ax.set_title("Vertical heat flux")
        ax.plot(fluxcond, z, label="Conductive")
        ax.plot(fluxconv, z, label="Convective")
        ax.plot(fluxtotal, z, label="Total")
        ax.legend()
        ax.set_ylabel('z')

        plt.tight_layout()
        plt.savefig(path.join(plot_dir, image_name))
        plt.close()


def plot_energy(data_dir, plot_dir):
    image_name = "energy.png"
    print(f'Plotting "{image_name}"...')
    with h5py.File(path.join(data_dir, 'analysis.h5'), mode='r') as file:
        t, _, _, _ = get_dims(file, 'E')

        KE = np.squeeze(get_field(file, 'E'))

        plots_shape = np.array((1, 1))
        plots_size_each = np.array((8, 4))
        fig = plt.figure(figsize=np.flip(plots_shape) * plots_size_each)

        ax = fig.add_subplot(*plots_shape, 1)
        ax.set_title("Kinetic energy as a function of time")
        ax.plot(t, KE)
        ax.set_ylabel('Energy')
        ax.set_xlabel('t')

        plt.tight_layout()
        plt.savefig(path.join(plot_dir, image_name))
        plt.close()


def plot_velocity_filters(data_dir, plot_dir):
    """Plot a snapshot in time of the velocities, against the low and high pass filtered versions"""
    image_name = "velocity_filters.png"
    print(f'Plotting "{image_name}"...')
    params = utils.read_params(data_dir)
    with h5py.File(path.join(data_dir, 'interp.h5'), mode='r') as file:

        t, x, y, z = get_dims(file, 'u')

        duration = min(params['duration'], t[-1])
        if duration < params['average_interval']: print('WARNING: averaging interval longer than simulation duration, averaging over entire duration...')
        tstart = duration - params['average_interval']
        timeframe_mask = np.logical_and(t >= tstart, t <= duration)

        t = t[timeframe_mask]

        u = get_field(file, 'u')[-1]
        v = get_field(file, 'v')[-1]
        w = get_field(file, 'w')[-1]

        wavelength = params["Lz"] / 2

        u_lowpass = filtering.kspace_lowpass(u, (0, 1, 2), (x, y, z), wavelength, interp=False)[:, params["resY"]//2]
        v_lowpass = filtering.kspace_lowpass(v, (0, 1, 2), (x, y, z), wavelength, interp=False)[:, params["resY"]//2]
        w_lowpass = filtering.kspace_lowpass(w, (0, 1, 2), (x, y, z), wavelength, interp=False)[:, params["resY"]//2]

        u_highpass = filtering.kspace_highpass(u, (0, 1, 2), (x, y, z), wavelength, interp=False)[:, params["resY"]//2]
        v_highpass = filtering.kspace_highpass(v, (0, 1, 2), (x, y, z), wavelength, interp=False)[:, params["resY"]//2]
        w_highpass = filtering.kspace_highpass(w, (0, 1, 2), (x, y, z), wavelength, interp=False)[:, params["resY"]//2]

        u = u[:, params["resY"]//2]
        v = v[:, params["resY"]//2]
        w = w[:, params["resY"]//2]

        plots_shape = np.array((3, 3))
        plots_size_each = np.array((8, 2))
        fig = plt.figure(figsize=np.flip(plots_shape) * plots_size_each)
        fig.suptitle(f'Snapshot of velocities at end of simulation, at y=Ly/2')

        ax = fig.add_subplot(*plots_shape, 1)
        ax.set_title("u")
        pcm = ax.pcolormesh(x, z, u.T, shading='nearest', cmap="CMRmap", label='u')
        ax.set_xlabel('x')
        ax.set_ylabel('z')

        ax = fig.add_subplot(*plots_shape, 2)
        ax.set_title("v")
        pcm = ax.pcolormesh(x, z, v.T, shading='nearest', cmap="CMRmap", label='v')
        ax.set_xlabel('x')
        ax.set_ylabel('z')

        ax = fig.add_subplot(*plots_shape, 3)
        ax.set_title("w")
        pcm = ax.pcolormesh(x, z, w.T, shading='nearest', cmap="CMRmap", label='w')
        ax.set_xlabel('x')
        ax.set_ylabel('z')

        ax = fig.add_subplot(*plots_shape, 4)
        ax.set_title("u lowpass")
        pcm = ax.pcolormesh(x, z, u_lowpass.T, shading='nearest', cmap="CMRmap", label='u')
        ax.set_xlabel('x')
        ax.set_ylabel('z')

        ax = fig.add_subplot(*plots_shape, 5)
        ax.set_title("v lowpass")
        pcm = ax.pcolormesh(x, z, v_lowpass.T, shading='nearest', cmap="CMRmap", label='v')
        ax.set_xlabel('x')
        ax.set_ylabel('z')

        ax = fig.add_subplot(*plots_shape, 6)
        ax.set_title("w lowpass")
        pcm = ax.pcolormesh(x, z, w_lowpass.T, shading='nearest', cmap="CMRmap", label='w')
        ax.set_xlabel('x')
        ax.set_ylabel('z')

        ax = fig.add_subplot(*plots_shape, 7)
        ax.set_title("u highpass")
        pcm = ax.pcolormesh(x, z, u_highpass.T, shading='nearest', cmap="CMRmap", label='u')
        ax.set_xlabel('x')
        ax.set_ylabel('z')

        ax = fig.add_subplot(*plots_shape, 8)
        ax.set_title("v highpass")
        pcm = ax.pcolormesh(x, z, v_highpass.T, shading='nearest', cmap="CMRmap", label='v')
        ax.set_xlabel('x')
        ax.set_ylabel('z')

        ax = fig.add_subplot(*plots_shape, 9)
        ax.set_title("w highpass")
        pcm = ax.pcolormesh(x, z, w_highpass.T, shading='nearest', cmap="CMRmap", label='w')
        ax.set_xlabel('x')
        ax.set_ylabel('z')

        plt.tight_layout()
        plt.savefig(path.join(plot_dir, image_name))
        plt.close()


def plot_momentum_terms_post(data_dir, plot_dir):
    image_name = "momentum_terms_post.png"
    print(f'Plotting "{image_name}"...')
    params = utils.read_params(data_dir)
    with h5py.File(path.join(data_dir, 'state.h5'), mode='r') as file:

        t, x, y, z = get_dims(file, 'u')

        duration = min(params['duration'], t[-1])
        if duration < params['average_interval']: print('WARNING: averaging interval longer than simulation duration, averaging over entire duration...')
        timeframe_mask = np.logical_and(t >= duration - params['average_interval'], t <= duration)

        t = t[timeframe_mask]

        print("  Reading file...")
        u = get_field(file, 'u')[timeframe_mask]
        v = get_field(file, 'v')[timeframe_mask]
        w = get_field(file, 'w')[timeframe_mask]

        # The x component terms
        print("  Calculating x terms...")
        coeff = np.sin(params["Theta"]) * params["Ta"]**0.5
        viscous_x = -np.gradient(np.gradient(u, z, axis=-1, edge_order=2), z, axis=-1, edge_order=2) / coeff
        coriolis_x = -v
        stress_x = np.mean(np.gradient(average_horizontal( (u - np.mean(u, axis=0, keepdims=True)) * (w - np.mean(w, axis=0, keepdims=True)) ), z, axis=-1, edge_order=2), axis=0) / coeff

        # The y component terms
        print("  Calculating y terms...")
        viscous_y = -np.gradient(np.gradient(v, z, axis=-1, edge_order=2), z, axis=-1, edge_order=2) / coeff
        coriolis_y = u
        stress_y = np.mean(np.gradient(average_horizontal( (v - np.mean(v, axis=0, keepdims=True)) * (w - np.mean(w, axis=0, keepdims=True)) ), z, axis=-1, edge_order=2), axis=0) / coeff

        # Averaging in time and horizontally in space...
        print("  Averaging...")
        # ... for x
        viscous_x_avg = np.mean(average_horizontal(viscous_x), axis=0)
        coriolis_x_avg = np.mean(average_horizontal(coriolis_x), axis=0)

        # ... and for y
        viscous_y_avg = np.mean(average_horizontal(viscous_y), axis=0)
        coriolis_y_avg = np.mean(average_horizontal(coriolis_y), axis=0)

        # Plot everything on two plots
        print("  Plotting...")
        plots_shape = np.array((2, 1))
        plots_size_each = np.array((8, 4))

        tstart = duration - params['average_interval']
        tend = duration
        fig = plt.figure(figsize=np.flip(plots_shape) * plots_size_each)
        fig.suptitle(
            "Terms of the averaged momentum equation\n"
            f"Averaged in t from {np.round(tstart, 2)} to {np.round(tend, 2)} viscous times\n"
            "All terms calculated in post-processing"
        )

        ax = fig.add_subplot(*plots_shape, 1)
        ax.set_title("x component")
        ax.axvline(0, lw=1, c='darkgray')
        ax.plot(viscous_x_avg, z, label="Viscous", c='green')
        ax.plot(-coriolis_x_avg, z, label="Mean flow", c='black')
        ax.plot(stress_x, z, label="Stress d/dz <uw>", c='red')
        ax.legend()
        ax.set_ylabel('z')

        ax = fig.add_subplot(*plots_shape, 2)
        ax.set_title("y component")
        ax.axvline(0, lw=1, c='darkgray')
        ax.plot(-viscous_y_avg, z, label="Viscous", c='green')
        ax.plot(coriolis_y_avg, z, label="Mean flow", c='black')
        ax.plot(-stress_y, z, label="Stress d/dz <vw>", c='red')
        ax.legend()
        ax.set_ylabel('z')

        plt.tight_layout()
        plt.savefig(path.join(plot_dir, image_name))
        plt.close()


def plot_momentum_terms_filtered(data_dir, plot_dir):
    image_name = "momentum_terms_filtered.png"
    print(f'Plotting "{image_name}"...')
    params = utils.read_params(data_dir)
    with h5py.File(path.join(data_dir, 'interp.h5'), mode='r') as file:

        t, x, y, z = get_dims(file, 'u')

        duration = min(params['duration'], t[-1])
        if duration < params['average_interval']: print('WARNING: averaging interval longer than simulation duration, averaging over entire duration...')
        timeframe_mask = np.logical_and(t >= duration - params['average_interval'], t <= duration)

        t = t[timeframe_mask]

        print("  Reading file...")
        u = get_field(file, 'u')[timeframe_mask]
        v = get_field(file, 'v')[timeframe_mask]
        w = get_field(file, 'w')[timeframe_mask]

        print("  Filtering velocity fields...")
        wavelength = params["Lz"] / 2
        u_low = filtering.kspace_lowpass(u, (1, 2, 3), (x, y, z), wavelength, interp=False)
        v_low = filtering.kspace_lowpass(v, (1, 2, 3), (x, y, z), wavelength, interp=False)
        w_low = filtering.kspace_lowpass(w, (1, 2, 3), (x, y, z), wavelength, interp=False)
        u_high = filtering.kspace_highpass(u, (1, 2, 3), (x, y, z), wavelength, interp=False)
        v_high = filtering.kspace_highpass(v, (1, 2, 3), (x, y, z), wavelength, interp=False)
        w_high = filtering.kspace_highpass(w, (1, 2, 3), (x, y, z), wavelength, interp=False)

        # The x component terms
        print("  Calculating x terms...")
        coeff = np.sin(params["Theta"]) * params["Ta"]**0.5
        viscous_x = -np.gradient(np.gradient(u, z, axis=-1, edge_order=2), z, axis=-1, edge_order=2) / coeff
        viscous_x_low = -np.gradient(np.gradient(u_low, z, axis=-1, edge_order=2), z, axis=-1, edge_order=2) / coeff
        viscous_x_high = -np.gradient(np.gradient(u_high, z, axis=-1, edge_order=2), z, axis=-1, edge_order=2) / coeff
        stress_x = np.mean(np.gradient(average_horizontal( (u - np.mean(u, axis=0, keepdims=True)) * (w - np.mean(w, axis=0, keepdims=True)) ), z, axis=-1, edge_order=2), axis=0) / coeff
        stress_x_low = np.mean(np.gradient(average_horizontal( (u_low - np.mean(u_low, axis=0, keepdims=True)) * (w_low - np.mean(w_low, axis=0, keepdims=True)) ), z, axis=-1, edge_order=2), axis=0) / coeff
        stress_x_high = np.mean(np.gradient(average_horizontal( (u_high - np.mean(u_high, axis=0, keepdims=True)) * (w_high - np.mean(w_high, axis=0, keepdims=True)) ), z, axis=-1, edge_order=2), axis=0) / coeff

        # The y component terms
        print("  Calculating y terms...")
        viscous_y = -np.gradient(np.gradient(v, z, axis=-1, edge_order=2), z, axis=-1, edge_order=2) / coeff
        viscous_y_low = -np.gradient(np.gradient(v_low, z, axis=-1, edge_order=2), z, axis=-1, edge_order=2) / coeff
        viscous_y_high = -np.gradient(np.gradient(v_high, z, axis=-1, edge_order=2), z, axis=-1, edge_order=2) / coeff
        stress_y = np.mean(np.gradient(average_horizontal( (v - np.mean(v, axis=0, keepdims=True)) * (w - np.mean(w, axis=0, keepdims=True)) ), z, axis=-1, edge_order=2), axis=0) / coeff
        stress_y_low = np.mean(np.gradient(average_horizontal( (v_low - np.mean(v_low, axis=0, keepdims=True)) * (w_low - np.mean(w_low, axis=0, keepdims=True)) ), z, axis=-1, edge_order=2), axis=0) / coeff
        stress_y_high = np.mean(np.gradient(average_horizontal( (v_high - np.mean(v_high, axis=0, keepdims=True)) * (w_high - np.mean(w_high, axis=0, keepdims=True)) ), z, axis=-1, edge_order=2), axis=0) / coeff

        # Averaging in time and horizontally in space...
        print("  Averaging...")
        # ... for x
        viscous_x_avg = np.mean(average_horizontal(viscous_x), axis=0)
        viscous_x_low_avg = np.mean(average_horizontal(viscous_x_low), axis=0)
        viscous_x_high_avg = np.mean(average_horizontal(viscous_x_high), axis=0)

        # ... and for y
        viscous_y_avg = np.mean(average_horizontal(viscous_y), axis=0)
        viscous_y_low_avg = np.mean(average_horizontal(viscous_y_low), axis=0)
        viscous_y_high_avg = np.mean(average_horizontal(viscous_y_high), axis=0)

        # Plot everything on two plots
        print("  Plotting...")
        plots_shape = np.array((2, 1))
        plots_size_each = np.array((8, 4))

        tstart = duration - params['average_interval']
        tend = duration
        fig = plt.figure(figsize=np.flip(plots_shape) * plots_size_each)
        fig.suptitle(
            "Terms of the averaged momentum equation\n"
            f"Averaged in t from {np.round(tstart, 2)} to {np.round(tend, 2)} viscous times\n"
            "All terms calculated in post-processing"
        )

        ax = fig.add_subplot(*plots_shape, 1)
        ax.set_title("x component")
        ax.axvline(0, lw=1, c='darkgray')
        ax.plot(viscous_x_avg, z, label="Viscous", c='green')
        ax.plot(viscous_x_low_avg, z, label="Viscous (lowpass)", lw=1, ls='--', c='green')
        ax.plot(viscous_x_high_avg, z, label="Viscous (highpass)", lw=1, ls=':', c='green')
        ax.plot(stress_x, z, label="Stress d/dz <uw>", c='red')
        ax.plot(stress_x_low, z, label="Stress (lowpass)", lw=1, ls='--', c='red')
        ax.plot(stress_x_high, z, label="Stress (highpass)", lw=1, ls=':', c='red')
        ax.legend()
        ax.set_ylabel('z')

        ax = fig.add_subplot(*plots_shape, 2)
        ax.set_title("y component")
        ax.axvline(0, lw=1, c='darkgray')
        ax.plot(-viscous_y_avg, z, label="Viscous", c='green')
        ax.plot(-viscous_y_low_avg, z, label="Viscous (lowpass)", lw=1, ls='--', c='green')
        ax.plot(-viscous_y_high_avg, z, label="Viscous (highpass)", lw=1, ls=':', c='green')
        ax.plot(-stress_y, z, label="Stress d/dz <vw>", c='red')
        ax.plot(-stress_y_low, z, label="Stress (lowpass)", lw=1, ls='--', c='red')
        ax.plot(-stress_y_high, z, label="Stress (highpass)", lw=1, ls=':', c='red')
        ax.legend()
        ax.set_ylabel('z')

        plt.tight_layout()
        plt.savefig(path.join(plot_dir, image_name))
        plt.close()


def plot_momentum_terms(data_dir, plot_dir):
    image_name = "momentum_terms.png"
    print(f'Plotting "{image_name}"...')
    params = utils.read_params(data_dir)
    with h5py.File(path.join(data_dir, 'analysis.h5'), mode='r') as file:

        t, _, _, z = get_dims(file, 'ViscousX')

        duration = min(params['duration'], t[-1])
        if duration < params['average_interval']: print('WARNING: averaging interval longer than simulation duration, averaging over entire duration...')
        tstart = duration - params['average_interval']
        tend = duration
        timeframe_mask = np.logical_and(t >= tstart, t <= duration)

        t = t[timeframe_mask]

        # Get the quantities from the analysis tasks
        viscous_x = np.mean(average_horizontal(get_field(file, 'ViscousX')[timeframe_mask]), axis=0)
        temporal_x = np.mean(average_horizontal(get_field(file, 'TemporalX')[timeframe_mask]), axis=0)
        coriolis_x = -np.mean(average_horizontal(get_field(file, 'MeanV')[timeframe_mask]), axis=0)
        stress_x = np.mean(average_horizontal(get_field(file, 'StressX')[timeframe_mask]), axis=0)

        viscous_y = np.mean(average_horizontal(get_field(file, 'ViscousY')[timeframe_mask]), axis=0)
        temporal_y = np.mean(average_horizontal(get_field(file, 'TemporalY')[timeframe_mask]), axis=0)
        coriolis_y = np.mean(average_horizontal(get_field(file, 'MeanU')[timeframe_mask]), axis=0)
        stress_y = np.mean(average_horizontal(get_field(file, 'StressY')[timeframe_mask]), axis=0)

        # Plot them on two different plots (one for x, one for y)
        plots_shape = np.array((2, 1))
        plots_size_each = np.array((8, 4))
        fig = plt.figure(figsize=np.flip(plots_shape) * plots_size_each)
        fig.suptitle(
            "Terms of the averaged momentum equation\n"
            f"Averaged in t from {np.round(tstart, 2)} to {np.round(tend, 2)} viscous times\n"
            "All terms calculated in dedalus\n"
            "Reynolds stresses calculated by assumption that these terms sum to exactly zero"
        )

        # Plotting in the same format as Currie & Tobias (2016) did.
        # This means flipping the coriolis term w.r.t. the other terms, as
        # they plotted the terms in an equation with the coriolis term
        # on the L.H.S. and everything else on the R.H.S.
        ax = fig.add_subplot(*plots_shape, 1)
        ax.set_title("x component")
        ax.plot(viscous_x, z, c='green', label="Viscous")
        ax.plot(temporal_x, z, c='orange', label="Temporal")
        ax.plot(stress_x, z, c='red', label="Stress")
        ax.plot(-coriolis_x, z, c='black', label="Coriolis")
        ax.legend()
        ax.set_ylabel('z')

        # Currie & Tobias (2016) also plotted the zonal direction the opposite way round,
        # as to make the coriolis term positive.
        ax = fig.add_subplot(*plots_shape, 2)
        ax.set_title("y component")
        ax.plot(-viscous_y, z, c='green', label="Viscous")
        ax.plot(-temporal_y, z, c='orange', label="Temporal")
        ax.plot(-stress_y, z, c='red', label="Stress")
        ax.plot(coriolis_y, z, c='black', label="Coriolis")
        ax.legend()
        ax.set_ylabel('z')

        plt.tight_layout()
        plt.savefig(path.join(plot_dir, image_name))
        plt.close()


def video(data_dir, plot_dir):
    print(f'Rendering video...')
    params = utils.read_params(data_dir)
    with h5py.File(path.join(data_dir, 'state.h5'), mode='r') as file:
        # Load datasets
        temp = average_zonal(get_field(file, 'T'))
        _, x, _, z = get_dims(file, 'T')

        params_string = utils.create_params_string(params)
        fig = plt.figure(figsize=utils.calc_plot_size(params), dpi=100)
        quad = plt.pcolormesh(x, z, temp[-1].T, shading='nearest', cmap="coolwarm")
        plt.colorbar()
        def animate(frame):
            # For some reason, matplotlib ends up getting the x and y axes the wrong way round,
            # so I just took the transpose of each frame to 'fix' it.
            quad.set_array(frame.T)
        plt.xlabel('x')
        plt.ylabel('z')
        plt.title(params_string, fontsize=9)
        s_per_visc_time = 60
        animation = ani.FuncAnimation(fig, animate, frames=temp, interval=params['timestep_analysis']*s_per_visc_time*1000)
        animation.save(path.join(plot_dir, 'video.mp4'))
        plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide one argument: The file path to the directory to read the analysis from.")
        exit(1)
    data_dir = sys.argv[1]
    plot_dir = sys.argv[2] if len(sys.argv) >= 3 else path.join(data_dir, "plots/")

    try:
        mkdir(plot_dir)
    except FileExistsError:
        pass

    plot_velocities(data_dir, plot_dir)
    plot_temperature(data_dir, plot_dir)
    plot_heat_flux_z(data_dir, plot_dir)
    plot_energy(data_dir, plot_dir)
    # plot_momentum_terms(data_dir, plot_dir)
    plot_momentum_terms_post(data_dir, plot_dir)
    plot_momentum_terms_filtered(data_dir, plot_dir)
    plot_velocity_filters(data_dir, plot_dir)
    # video(data_dir, plot_dir)
