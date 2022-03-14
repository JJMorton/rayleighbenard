#!/usr/bin/env python3

import h5py
import numpy as np
# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from os import path
from os import mkdir
import sys
import dedalus.public as de

import filtering
import utils


def get_field(file, fieldname):
    """Get a field from the .h5 file, with dimensions (t, x, y, z)"""
    arr = np.array(file['tasks'][fieldname])
    if len(arr.shape) == 3:
        # 2d sim, add dummy y dimension
        arr = arr[:, :, np.newaxis, :]
    return arr

def get_dims_interp(file):
    t = np.array(file['scales']['t'])
    x = np.array(file['scales']['x'])
    y = np.array(file['scales']['y'])
    z = np.array(file['scales']['z'])
    return t, x, y, z

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
        raise Exception("Attempt to zonally average array with {} dimensions".format(dims))
    return np.mean(arr, axis=2)

def average_horizontal(arr):
    """Average the field horizonally: (t, x, y, z) --> (t, z)"""
    dims = len(arr.shape)
    if dims != 4:
        raise Exception("Attempt to horizontally average array with {} dimensions".format(dims))
    return np.mean(np.mean(arr, axis=1), axis=1)

def calc_momentum_terms(params, x, y, z, u, v, w):
    coeff = np.sin(params["Theta"]) * params["Ta"]**0.5

    print("    Viscous X")
    viscous_x = -np.gradient(np.gradient(u, z, axis=-1, edge_order=2), z, axis=-1, edge_order=2) / coeff
    viscous_x_avg = np.mean(average_horizontal(viscous_x), axis=0)
    del viscous_x
    print("    Viscous Y")
    viscous_y = -np.gradient(np.gradient(v, z, axis=-1, edge_order=2), z, axis=-1, edge_order=2) / coeff
    viscous_y_avg = np.mean(average_horizontal(viscous_y), axis=0)
    del viscous_y

    print("    Time averages")
    u_avgt = np.mean(u, axis=0)
    v_avgt = np.mean(v, axis=0)
    w_avgt = np.mean(w, axis=0)

    print("    Coriolis X")
    coriolis_x_avg = -np.mean(np.mean(v_avgt, axis=0), axis=0)
    print("    Coriolis Y")
    coriolis_y_avg = np.mean(np.mean(u_avgt, axis=0), axis=0)

    print("    RS X")
    stress_x = np.gradient(np.mean(average_horizontal((u - u_avgt) * (w - w_avgt)), axis=0), z, axis=-1, edge_order=2) / coeff
    print("    RS Y")
    stress_y = np.mean(np.gradient(average_horizontal((v - v_avgt) * (w - w_avgt)), z, axis=-1, edge_order=2), axis=0) / coeff

    del u_avgt
    del v_avgt
    del w_avgt

    return viscous_x_avg, viscous_y_avg, coriolis_x_avg, coriolis_y_avg, stress_x, stress_y


def plot_velocities(data_dir, plot_dir):
    """Plot the time averaged velocities"""
    image_name = "velocity.png"
    print('Plotting "{}"...'.format(image_name))
    params = utils.read_params(data_dir)
    filepath = path.join(data_dir, 'state.h5')
    if not path.exists(filepath):
        print("Plotting '{}' requires '{}'".format(image_name, filepath))
        return

    with h5py.File(filepath, mode='r') as file:

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
        fig = plt.figure(figsize=plots_shape[::-1] * plots_size_each)
        fig.suptitle('Averaged zonally and in time from {:.2f} to {:.2f} viscous times'.format(tstart, duration))

        ax = fig.add_subplot(*plots_shape, 1)
        ax.set_title("Time averaged u")
        pcm = ax.pcolormesh(x, z, u_avgt.T, shading='nearest', cmap="CMRmap", label='<u>')
        fig.colorbar(pcm, ax=ax)
        ax.set_xlabel('x')
        ax.set_ylabel('z')

        ax = fig.add_subplot(*plots_shape, 2)
        ax.set_title("Time averaged v")
        pcm = ax.pcolormesh(x, z, v_avgt.T, shading='nearest', cmap="CMRmap", label='<v>')
        fig.colorbar(pcm, ax=ax)
        ax.set_xlabel('x')
        ax.set_ylabel('z')

        ax = fig.add_subplot(*plots_shape, 3)
        ax.set_title("Time averaged w")
        pcm = ax.pcolormesh(x, z, w_avgt.T, shading='nearest', cmap="CMRmap", label='<w>')
        fig.colorbar(pcm, ax=ax)
        ax.set_xlabel('x')
        ax.set_ylabel('z')

        fig.set_tight_layout(True)
        plt.savefig(path.join(plot_dir, image_name))
        plt.close()


def plot_temperature(data_dir, plot_dir):
    image_name = "temperature.png"
    print('Plotting "{}"...'.format(image_name))
    params = utils.read_params(data_dir)
    filepath = path.join(data_dir, 'analysis.h5')
    filepath2 = path.join(data_dir,'state.h5')
    if not path.exists(filepath):
        print("Plotting '{}' requires '{}'".format(image_name, filepath))
        return

    with h5py.File(filepath2, mode='r') as file:

        t, x, y, z = get_dims(file, 'u')

    with h5py.File(filepath, mode='r') as file:

        Ttop = np.squeeze(get_field(file, 'Ttop'))[-1]
        Tmid = np.squeeze(get_field(file, 'Tmid'))[-1]

        plots_shape = np.array((1, 2))
        plots_size_each = np.array((8, 8))
        fig = plt.figure(figsize=plots_shape[::-1] * plots_size_each)

        ax = fig.add_subplot(*plots_shape, 1)
        ax.set_title("Top view")
        pcm = ax.pcolormesh(x, y, Ttop.T, shading='nearest', cmap="CMRmap", label='Ttop')
        fig.colorbar(pcm, ax=ax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect(1)

        ax = fig.add_subplot(*plots_shape, 2)
        ax.set_title("Side view")
        pcm = ax.pcolormesh(x, z, Tmid.T, shading='nearest', cmap="CMRmap", label='Tmid')
        fig.colorbar(pcm, ax=ax)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_aspect(1)

        fig.set_tight_layout(True)
        plt.savefig(path.join(plot_dir, image_name))
        plt.close()


def plot_heat_flux_z(data_dir, plot_dir):
    image_name = "heat_flux_z.png"
    print('Plotting "{}"...'.format(image_name))
    params = utils.read_params(data_dir)
    filepath = path.join(data_dir, 'analysis.h5')
    if not path.exists(filepath):
        print("Plotting '{}' requires '{}'".format(image_name, filepath))
        return

    with h5py.File(filepath, mode='r') as file:

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
        fig = plt.figure(figsize=plots_shape[::-1] * plots_size_each)
        fig.suptitle('Averaged from {:.2f} to {:.2f} viscous times'.format(tstart, duration))

        ax = fig.add_subplot(*plots_shape, 1)
        ax.set_title("Vertical heat flux")
        ax.plot(fluxcond, z, label="Conductive")
        ax.plot(fluxconv, z, label="Convective")
        ax.plot(fluxtotal, z, label="Total")
        ax.legend()
        ax.set_ylabel('z')

        fig.set_tight_layout(True)
        plt.savefig(path.join(plot_dir, image_name))
        plt.close()


def plot_energy(data_dir, plot_dir):
    image_name = "energy.png"
    print('Plotting "{}"...'.format(image_name))
    filepath = path.join(data_dir, 'analysis.h5')
    if not path.exists(filepath):
        print("Plotting '{}' requires '{}'".format(image_name, filepath))
        return

    with h5py.File(filepath, mode='r') as file:
        t, _, _, _ = get_dims(file, 'E')

        KE = np.squeeze(get_field(file, 'E'))

        plots_shape = np.array((1, 1))
        plots_size_each = np.array((8, 4))
        fig = plt.figure(figsize=plots_shape[::-1] * plots_size_each)

        ax = fig.add_subplot(*plots_shape, 1)
        ax.set_title("Kinetic energy as a function of time")
        ax.plot(t, KE)
        ax.set_ylabel('Energy')
        ax.set_xlabel('t')

        fig.set_tight_layout(True)
        plt.savefig(path.join(plot_dir, image_name))
        plt.close()


# TODO: FIX THIS MEMORY USAGE
def plot_velocity_filters(data_dir, plot_dir):
    """Plot a snapshot in time of the velocities, against the low and high pass filtered versions"""
    image_name = "velocity_filters.png"
    print('Plotting "{}"...'.format(image_name))
    params = utils.read_params(data_dir)
    filepath1 = path.join(data_dir, 'interp_u.h5')
    filepath2 = path.join(data_dir, 'interp_v.h5')
    filepath3 = path.join(data_dir, 'interp_w.h5')
    if not path.exists(filepath1):
        print("Plotting '{}' requires '{}'".format(image_name, filepath1))
        return
    if not path.exists(filepath2):
        print("Plotting '{}' requires '{}'".format(image_name, filepath2))
        return
    if not path.exists(filepath3):
        print("Plotting '{}' requires '{}'".format(image_name, filepath3))
        return

    with h5py.File(filepath1, mode='r') as file:

        t, x, y, z = get_dims_interp(file)
        is3D = y is not None

        duration = min(params['duration'], t[-1])
        if duration < params['average_interval']: print('WARNING: averaging interval longer than simulation duration, averaging over entire duration...')
        tstart = duration - params['average_interval']

        # Correct the order of the axes after reading in the fields
        u = get_field(file, 'u')[-1]
        u_lowpass = get_field(file, 'u_lowpass')[-1]
        u_highpass = get_field(file, 'u_highpass')[-1]

    with h5py.File(filepath2, mode='r') as file:
        v = get_field(file, 'v')[-1]
        v_lowpass = get_field(file, 'v_lowpass')[-1]
        v_highpass = get_field(file, 'v_highpass')[-1]
    
    with h5py.File(filepath3, mode='r') as file:
        w = get_field(file, 'w')[-1]
        w_lowpass = get_field(file, 'w_lowpass')[-1]
        w_highpass = get_field(file, 'w_highpass')[-1]

    indices = (0, 1, 2) if is3D else (0, 1)
    bases = (x, y, z) if is3D else (x, z)

    # In 3D, take a slice at constant y so we can plot in 2D
    if is3D:
        u_highpass = u_highpass[:, params["resY"]//2]
        v_highpass = v_highpass[:, params["resY"]//2]
        w_highpass = w_highpass[:, params["resY"]//2]
        u_lowpass = u_lowpass[:, params["resY"]//2]
        v_lowpass = v_lowpass[:, params["resY"]//2]
        w_lowpass = w_lowpass[:, params["resY"]//2]
        u = u[:, params["resY"]//2]
        v = v[:, params["resY"]//2]
        w = w[:, params["resY"]//2]

    plots_shape = np.array((3, 3))
    plots_size_each = np.array((8, 4))
    fig = plt.figure(figsize=plots_shape * plots_size_each)
    if is3D:
        fig.suptitle('Snapshot of velocities at t={:.2f} and y=Ly/2'.format(duration))
    else:
        fig.suptitle('Snapshot of velocities at t={:.2f}')

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

    fig.set_tight_layout(True)
    plt.savefig(path.join(plot_dir, image_name))
    plt.close()


def plot_momentum_terms_post(data_dir, plot_dir):
    image_name = "momentum_terms_post.png"
    print('Plotting "{}"...'.format(image_name))
    params = utils.read_params(data_dir)
    filepath = path.join(data_dir, 'state.h5')
    if not path.exists(filepath):
        print("Plotting '{}' requires '{}'".format(image_name, filepath))
        return

    with h5py.File(filepath, mode='r') as file:

        t, x, y, z = get_dims(file, 'u')

        if y is None:
            print("Plotting momentum terms only has support for 3D right now.")
            return

        duration = min(params['duration'], t[-1])
        if duration < params['average_interval']: print('WARNING: averaging interval longer than simulation duration, averaging over entire duration...')
        timeframe_mask = np.logical_and(t >= duration - params['average_interval'], t <= duration)

        t = t[timeframe_mask]

        print("  Reading file...")
        u = get_field(file, 'u')[timeframe_mask]
        v = get_field(file, 'v')[timeframe_mask]
        w = get_field(file, 'w')[timeframe_mask]

        viscous_x, viscous_y, coriolis_x, coriolis_y, rs_x, rs_y = calc_momentum_terms(params, x, y, z, u, v, w)
        del u
        del v
        del w

        # Plot everything on two plots
        print("  Plotting...")
        plots_shape = np.array((2, 1))
        plots_size_each = np.array((8, 4))

        tstart = duration - params['average_interval']
        tend = duration
        fig = plt.figure(figsize=plots_shape[::-1] * plots_size_each)
        fig.suptitle(
            "Terms of the averaged momentum equation\n" +
            "Averaged in t from {:.2f} to {:.2f} viscous times\n".format(tstart, tend) +
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

        fig.set_tight_layout(True)
        plt.savefig(path.join(plot_dir, image_name))
        plt.close()


def plot_momentum_terms_filtered(data_dir, plot_dir):
    image_name = "momentum_terms_filtered.png"
    print('Plotting "{}"...'.format(image_name))
    params = utils.read_params(data_dir)
    filepath1 = path.join(data_dir, 'interp_u.h5')
    filepath2 = path.join(data_dir, 'interp_v.h5')
    filepath3 = path.join(data_dir, 'interp_w.h5')
    if not path.exists(filepath1):
        print("Plotting '{}' requires '{}'".format(image_name, filepath1))
        return
    if not path.exists(filepath2):
        print("Plotting '{}' requires '{}'".format(image_name, filepath2))
        return
    if not path.exists(filepath3):
        print("Plotting '{}' requires '{}'".format(image_name, filepath3))
        return

    print("  Reading lowpassed fields from files...")
    with h5py.File(filepath1, mode='r') as file:
        t, x, y, z = get_dims_interp(file)
        u_low = get_field(file, 'u_lowpass')
    with h5py.File(filepath2, mode='r') as file:
        v_low = get_field(file, 'v_lowpass')
    with h5py.File(filepath3, mode='r') as file:
        w_low = get_field(file, 'w_lowpass')

    print("  Calculating lowpass filtered terms...")
    viscous_x_low, viscous_y_low, _, _, rs_x_low, rs_y_low = calc_momentum_terms(params, x, y, z, u_low, v_low, w_low)
    del u_low
    del v_low
    del w_low

    print("  Reading highpassed fields from files...")
    with h5py.File(filepath1, mode='r') as file:
        u_high = get_field(file, 'u_highpass')
    with h5py.File(filepath2, mode='r') as file:
        v_high = get_field(file, 'v_highpass')
    with h5py.File(filepath3, mode='r') as file:
        w_high = get_field(file, 'w_highpass')

    print("  Calculating highpass filtered terms...")
    viscous_x_high, viscous_y_high, _, _, rs_x_high, rs_y_high = calc_momentum_terms(params, x, y, z, u_high, v_high, w_high)
    del u_high
    del v_high
    del w_high

    # Plot everything on two plots
    print("  Plotting...")
    plots_shape = np.array((2, 1))
    plots_size_each = np.array((8, 4))

    tstart = t[0]
    tend = t[-1]
    fig = plt.figure(figsize=plots_shape[::-1] * plots_size_each)
    fig.suptitle("Terms of the averaged momentum equation\nAveraged in t from {:.2f} to {:.2f} viscous times\nAll terms calculated in post-processing".format(tstart, tend))

    ax = fig.add_subplot(*plots_shape, 1)
    ax.set_title("x component")
    ax.axvline(0, lw=1, c='darkgray')
    # ax.plot(-coriolis_x, z, label="Mean flow", c='lightgray', lw=4)
    # ax.plot(viscous_x, z, label="Viscous", c='green')
    ax.plot(viscous_x_low, z, label="Viscous (lowpass)", lw=1, ls='--', c='green')
    ax.plot(viscous_x_high, z, label="Viscous (highpass)", lw=1, ls=':', c='green')
    # ax.plot(rs_x, z, label="Stress d/dz <uw>", c='red')
    ax.plot(rs_x_low, z, label="Stress (lowpass)", lw=1, ls='--', c='red')
    ax.plot(rs_x_high, z, label="Stress (highpass)", lw=1, ls=':', c='red')
    ax.legend()
    ax.set_ylabel('z')

    ax = fig.add_subplot(*plots_shape, 2)
    ax.set_title("y component")
    ax.axvline(0, lw=1, c='darkgray')
    # ax.plot(coriolis_y, z, label="Mean flow", c='lightgray', lw=4)
    # ax.plot(-viscous_y, z, label="Viscous", c='green')
    ax.plot(-viscous_y_low, z, label="Viscous (lowpass)", lw=1, ls='--', c='green')
    ax.plot(-viscous_y_high, z, label="Viscous (highpass)", lw=1, ls=':', c='green')
    # ax.plot(-rs_y, z, label="Stress d/dz <vw>", c='red')
    ax.plot(-rs_y_low, z, label="Stress (lowpass)", lw=1, ls='--', c='red')
    ax.plot(-rs_y_high, z, label="Stress (highpass)", lw=1, ls=':', c='red')
    ax.legend()
    ax.set_ylabel('z')

    fig.set_tight_layout(True)
    plt.savefig(path.join(plot_dir, image_name))
    plt.close()

    
def video(data_dir, plot_dir):
    print('Rendering video...')
    params = utils.read_params(data_dir)
    filepath = path.join(data_dir, 'state.h5')
    if not path.exists(filepath):
        print("Creating video requires '{}'".format(filepath))
        return

    with h5py.File(filepath, mode='r') as file:
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
    plot_velocity_filters(data_dir, plot_dir)
    plot_momentum_terms_post(data_dir, plot_dir)
    # plot_momentum_terms_filtered(data_dir, plot_dir)
    # plot_momentum_terms(data_dir, plot_dir)
    # video(data_dir, plot_dir)

    print("Done.")

