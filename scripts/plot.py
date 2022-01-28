#!/usr/bin/env python3

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from os import path
from os import mkdir
import sys

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
        dims.insert(2, np.array([0]))
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
    image_name = "velocity.jpg"
    print(f'Plotting "{image_name}"...')
    params = utils.read_params(data_dir)
    with h5py.File(path.join(data_dir, 'state.h5'), mode='r') as file:

        t, x, _, z = get_dims(file, 'u')

        duration = min(params['duration'], t[-1])
        if duration < params['average_interval']: print('WARNING: averaging interval longer than simulation duration, averaging over entire duration...')
        tstart = duration - params['average_interval']
        timeframe_mask = np.logical_and(t >= tstart, t <= duration)

        t = t[timeframe_mask]

        u = average_zonal(get_field(file, 'u'))[timeframe_mask]
        v = average_zonal(get_field(file, 'v'))[timeframe_mask]
        w = average_zonal(get_field(file, 'w'))[timeframe_mask]

        u_avgt = np.mean(u, axis=0)
        v_avgt = np.mean(v, axis=0)
        w_avgt = np.mean(w, axis=0)

        plots_shape = np.array((2, 2))
        plots_size_each = np.array((8, 4))
        fig = plt.figure(figsize=np.flip(plots_shape) * plots_size_each)
        fig.suptitle(f'Averaged from {np.round(tstart, 2)} to {np.round(duration, 2)} viscous times')

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


def plot_heat_flux_z(data_dir, plot_dir):
    image_name = "heat_flux_z.jpg"
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
    image_name = "energy.jpg"
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


# Probably not very useful now, just left here for reference
def plot_momentum_x_terms(data_dir, plot_dir):
    image_name = "momentum_x_terms.jpg"
    print(f'Plotting "{image_name}"...')
    params = utils.read_params(data_dir)
    with h5py.File(path.join(data_dir, 'state.h5'), mode='r') as file:

        t, x, _, z = get_dims(file, 'u')

        duration = min(params['duration'], t[-1])
        if duration < params['average_interval']: print('WARNING: averaging interval longer than simulation duration, averaging over entire duration...')
        timeframe_mask = np.logical_and(t >= duration - params['average_interval'], t <= duration)

        t = t[timeframe_mask]

        u = get_field(file, 'u')[timeframe_mask]
        v = get_field(file, 'v')[timeframe_mask]
        w = get_field(file, 'w')[timeframe_mask]
        u_dz = get_field(file, 'u_dz')[timeframe_mask]
        u_dx = get_field(file, 'u_dx')[timeframe_mask]
        u_dt = get_field(file, 'u_dt')[timeframe_mask]
        u_dz2 = get_field(file, 'u_dz2')[timeframe_mask]

        temporal = u_dt
        temporal_np = np.gradient(u, t, axis=0, edge_order=2)
        viscous = -u_dz2
        viscous_np = -np.gradient(np.gradient(u, z, axis=-1, edge_order=2), z, axis=-1, edge_order=2)
        inertial = w * u_dz + u * u_dx
        inertial_np = w * np.gradient(u, z, axis=-1, edge_order=2) + u * np.gradient(u, x, axis=1, edge_order=2)
        coriolis = v * np.sin(params["Theta"]) * params["Ta"]**0.5

        snapshot_time_index = -1
        snapshot_time = t[snapshot_time_index]
        temporal_avgx = average_horizontal(temporal)[snapshot_time_index]
        temporal_np_avgx = average_horizontal(temporal_np)[snapshot_time_index]
        viscous_avgx = average_horizontal(viscous)[snapshot_time_index]
        viscous_np_avgx = average_horizontal(viscous_np)[snapshot_time_index]
        inertial_avgx = average_horizontal(inertial)[snapshot_time_index]
        inertial_np_avgx = average_horizontal(inertial_np)[snapshot_time_index]
        coriolis_avgx = average_horizontal(coriolis)[snapshot_time_index]
        total_np_avgx = temporal_np_avgx + viscous_np_avgx + inertial_np_avgx - coriolis_avgx
        total_avgx = temporal_avgx + viscous_avgx + inertial_avgx - coriolis_avgx

        stress_avgxt = np.mean(np.gradient(average_horizontal( (u - np.mean(u, axis=0, keepdims=True)) * (w - np.mean(w, axis=0, keepdims=True)) ), z, axis=-1, edge_order=2), axis=0)
        temporal_avgxt = np.mean(average_horizontal(temporal), axis=0)
        temporal_np_avgxt = np.mean(average_horizontal(temporal_np), axis=0)
        viscous_avgxt = np.mean(average_horizontal(viscous), axis=0)
        viscous_np_avgxt = np.mean(average_horizontal(viscous_np), axis=0)
        coriolis_avgxt = np.mean(average_horizontal(coriolis), axis=0)
        total_np_avgxt = temporal_np_avgxt + viscous_np_avgxt + stress_avgxt - coriolis_avgxt
        total_avgxt = temporal_avgxt + viscous_avgxt + stress_avgxt - coriolis_avgxt

        plots_shape = np.array((2, 1))
        plots_size_each = np.array((8, 4))

        tstart = duration - params['average_interval']
        tend = duration
        fig = plt.figure(figsize=np.flip(plots_shape) * plots_size_each)
        fig.suptitle("Dotted lines calculated in post (where possible),\nsolid lines are equivalents using dedalus")

        ax = fig.add_subplot(*plots_shape, 1)
        ax.set_title("Terms of the momentum equation (x component), averaged in x at t={0:.2f}".format(snapshot_time))
        ax.plot(temporal_np_avgx, z, c='orange', ls=':')
        ax.plot(temporal_avgx, z, label="temporal", c='orange', lw=1)
        ax.plot(viscous_np_avgx, z, c='green', ls=':')
        ax.plot(viscous_avgx, z, label="viscous", c='green', lw=1)
        ax.plot(inertial_np_avgx, z, c='red', ls=':')
        ax.plot(inertial_avgx, z, label="inertial", c='red', lw=1)
        ax.plot(coriolis_avgx, z, label="coriolis", c='black', lw=1)
        ax.plot(total_np_avgx, z, c='lightgray', ls=':')
        ax.plot(total_avgx, z, label="total", c='lightgray', lw=1)
        ax.legend()
        ax.set_ylabel('z')

        ax = fig.add_subplot(*plots_shape, 2)
        ax.set_title(f'Terms of the time-averaged momentum equation (x component), averaged in x\nAveraged in t from {np.round(tstart, 2)} to {np.round(tend, 2)} viscous times')
        ax.plot(temporal_np_avgxt, z, c='orange', ls=':')
        ax.plot(temporal_avgxt, z, label="temporal", c='orange', lw=1)
        ax.plot(viscous_np_avgxt, z, c='green', ls=':')
        ax.plot(viscous_avgxt, z, label="viscous", c='green', lw=1)
        ax.plot(stress_avgxt, z, label="stress (post)", c='red', lw=1)
        ax.plot(coriolis_avgxt, z, label="coriolis", c='black', lw=1)
        ax.plot(total_np_avgxt, z, c='lightgray', ls=':')
        ax.plot(total_avgxt, z, label="total", c='lightgray', lw=1)
        ax.legend()
        ax.set_ylabel('z')

        plt.tight_layout()
        plt.savefig(path.join(plot_dir, image_name))
        plt.close()


# Probably not very useful now, just left here for reference
def plot_momentum_y_terms(data_dir, plot_dir):
    image_name = "momentum_y_terms.jpg"
    print(f'Plotting "{image_name}"...')
    params = utils.read_params(data_dir)
    with h5py.File(path.join(data_dir, 'state.h5'), mode='r') as file:

        t, x, _, z = get_dims(file, 'u')

        duration = min(params['duration'], t[-1])
        if duration < params['average_interval']: print('WARNING: averaging interval longer than simulation duration, averaging over entire duration...')
        timeframe_mask = np.logical_and(t >= duration - params['average_interval'], t <= duration)

        t = t[timeframe_mask]

        u = get_field(file, 'u')[timeframe_mask]
        v = get_field(file, 'v')[timeframe_mask]
        w = get_field(file, 'w')[timeframe_mask]
        v_dz = get_field(file, 'v_dz')[timeframe_mask]
        v_dx = get_field(file, 'v_dx')[timeframe_mask]
        v_dt = get_field(file, 'v_dt')[timeframe_mask]
        v_dz2 = get_field(file, 'v_dz2')[timeframe_mask]

        temporal = -v_dt
        temporal_np = -np.gradient(v, t, axis=0, edge_order=2)
        viscous = v_dz2
        viscous_np = np.gradient(np.gradient(v, z, axis=-1, edge_order=2), z, axis=-1, edge_order=2)
        inertial = -(w * v_dz + u * v_dx)
        inertial_np = -(w * np.gradient(v, z, axis=-1, edge_order=2) + u * np.gradient(v, x, axis=1, edge_order=2))
        coriolis = u * np.sin(params["Theta"]) * params["Ta"]**0.5

        snapshot_time_index = -1
        snapshot_time = t[snapshot_time_index]
        temporal_avgx = average_horizontal(temporal)[snapshot_time_index]
        temporal_np_avgx = average_horizontal(temporal_np)[snapshot_time_index]
        viscous_avgx = average_horizontal(viscous)[snapshot_time_index]
        viscous_np_avgx = average_horizontal(viscous_np)[snapshot_time_index]
        inertial_avgx = average_horizontal(inertial)[snapshot_time_index]
        inertial_np_avgx = average_horizontal(inertial_np)[snapshot_time_index]
        coriolis_avgx = average_horizontal(coriolis)[snapshot_time_index]
        total_np_avgx = temporal_np_avgx + viscous_np_avgx + inertial_np_avgx - coriolis_avgx
        total_avgx = temporal_avgx + viscous_avgx + inertial_avgx - coriolis_avgx

        stress_avgxt = -np.mean(np.gradient(average_horizontal( (v - np.mean(v, axis=0, keepdims=True)) * (w - np.mean(w, axis=0, keepdims=True)) ), z, axis=-1, edge_order=2), axis=0)
        temporal_avgxt = np.mean(average_horizontal(temporal), axis=0)
        temporal_np_avgxt = np.mean(average_horizontal(temporal_np), axis=0)
        viscous_avgxt = np.mean(average_horizontal(viscous), axis=0)
        viscous_np_avgxt = np.mean(average_horizontal(viscous_np), axis=0)
        coriolis_avgxt = np.mean(average_horizontal(coriolis), axis=0)
        total_np_avgxt = temporal_np_avgxt + viscous_np_avgxt + stress_avgxt - coriolis_avgxt
        total_avgxt = temporal_avgxt + viscous_avgxt + stress_avgxt - coriolis_avgxt

        plots_shape = np.array((2, 1))
        plots_size_each = np.array((8, 4))

        tstart = duration - params['average_interval']
        tend = duration
        fig = plt.figure(figsize=np.flip(plots_shape) * plots_size_each)
        fig.suptitle("Dotted lines calculated in post (where possible),\nsolid lines are equivalents using dedalus")

        ax = fig.add_subplot(*plots_shape, 1)
        ax.set_title("Terms of the momentum equation (y component), averaged in x at t={0:.2f}".format(snapshot_time))
        ax.plot(temporal_np_avgx, z, c='orange', ls=':')
        ax.plot(temporal_avgx, z, label="temporal", c='orange', lw=1)
        ax.plot(viscous_np_avgx, z, c='green', ls=':')
        ax.plot(viscous_avgx, z, label="viscous", c='green', lw=1)
        ax.plot(inertial_np_avgx, z, c='red', ls=':')
        ax.plot(inertial_avgx, z, label="inertial", c='red', lw=1)
        ax.plot(coriolis_avgx, z, label="coriolis", c='black', lw=1)
        ax.plot(total_np_avgx, z, c='lightgray', ls=':')
        ax.plot(total_avgx, z, label="total", c='lightgray', lw=1)
        ax.legend()
        ax.set_ylabel('z')

        ax = fig.add_subplot(*plots_shape, 2)
        ax.set_title(f'Terms of the time-averaged momentum equation (y component), averaged in x\nAveraged in t from {np.round(tstart, 2)} to {np.round(tend, 2)} viscous times')
        ax.plot(temporal_np_avgxt, z, c='orange', ls=':')
        ax.plot(temporal_avgxt, z, label="temporal", c='orange', lw=1)
        ax.plot(viscous_np_avgxt, z, c='green', ls=':')
        ax.plot(viscous_avgxt, z, label="viscous", c='green', lw=1)
        ax.plot(stress_avgxt, z, label="stress (post)", c='red', lw=1)
        ax.plot(coriolis_avgxt, z, label="coriolis", c='black', lw=1)
        ax.plot(total_np_avgxt, z, c='lightgray', ls=':')
        ax.plot(total_avgxt, z, label="total", c='lightgray', lw=1)
        ax.legend()
        ax.set_ylabel('z')

        plt.tight_layout()
        plt.savefig(path.join(plot_dir, image_name))
        plt.close()


def plot_temperature(data_dir, plot_dir):
    image_name = "temperature.jpg"
    print(f'Plotting "{image_name}"...')
    params = utils.read_params(data_dir)
    with h5py.File(path.join(data_dir, 'state.h5'), mode='r') as file:

        t, x, _, z = get_dims(file, 'u')

        duration = min(params['duration'], t[-1])
        if duration < params['average_interval']: print('WARNING: averaging interval longer than simulation duration, averaging over entire duration...')
        tstart = duration - params['average_interval']
        timeframe_mask = np.logical_and(t >= tstart, t <= duration)

        t = t[timeframe_mask]

        T = average_zonal(get_field(file, 'T'))[timeframe_mask]
        T_avgt = np.mean(T, axis=0)

        plots_shape = np.array((1, 1))
        plots_size_each = np.array((8, 4))
        fig = plt.figure(figsize=np.flip(plots_shape) * plots_size_each)
        fig.suptitle(f'Averaged from {np.round(tstart, 2)} to {np.round(duration, 2)} viscous times')

        ax = fig.add_subplot(*plots_shape, 1)
        ax.set_title("Time averaged T")
        pcm = ax.pcolormesh(x, z, T_avgt.T, shading='nearest', cmap="CMRmap", label='<T>')
        fig.colorbar(pcm, ax=ax)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_aspect(1)

        plt.tight_layout()
        plt.savefig(path.join(plot_dir, image_name))
        plt.close()


def plot_averaged_momentum_eq(data_dir, plot_dir):
    image_name = "averaged_momentum_eq.jpg"
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
        fig.suptitle(f'Averaged from {np.round(tstart, 2)} to {np.round(tend, 2)} viscous times')

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

    plot_averaged_momentum_eq(data_dir, plot_dir)
    plot_velocities(data_dir, plot_dir)
    plot_momentum_x_terms(data_dir, plot_dir)
    plot_momentum_y_terms(data_dir, plot_dir)
    plot_heat_flux_z(data_dir, plot_dir)
    plot_energy(data_dir, plot_dir)
    plot_temperature(data_dir, plot_dir)
    # video(data_dir, plot_dir)
