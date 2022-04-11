#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from os import path
from os import mkdir
import sys

import utils

FILE_EXT = "pdf"
LATEX = True

def plot_stresses(csv_dir, plot_dir, z):
    image_name = "stresses.{}".format(FILE_EXT)
    print('Plotting "{}"...'.format(image_name))
    filepath = path.join(csv_dir, 'stresses.csv')
    if not path.exists(filepath):
        print("Plotting '{}' requires '{}'".format(image_name, filepath))
        return

    data = np.loadtxt(filepath, delimiter=',')
    stress_uw = data[:, 0]
    stress_vw = data[:, 1]
    stress_uv = data[:, 2]

    # Plot everything on two plots
    plots_shape = np.array((1, 1))
    plots_size_each = np.array((3.2, 2.7))

    fig = plt.figure(figsize=plots_shape[::-1] * plots_size_each)

    ax = fig.add_subplot(*plots_shape, 1)
    ax.axvline(0, lw=1, c='lightgray')
    ax.plot(stress_uw, z, label=r"$\langle uw \rangle$" if LATEX else "<uw>")
    ax.plot(stress_vw, z, label=r"$\langle vw \rangle$" if LATEX else "<vw>")
    ax.plot(stress_uv, z, label=r"$\langle uv \rangle$" if LATEX else "<uw>")
    ax.set_ylabel(r'$z$' if LATEX else 'z')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), shadow=False, ncol=3, columnspacing=1, frameon=False)

    fig.set_tight_layout(True)
    plt.savefig(path.join(plot_dir, image_name))
    plt.close()


def plot_velocities(csv_dir, plot_dir, x, z):
    """Plot the time averaged velocities"""
    image_name = "velocity.{}".format(FILE_EXT)
    print('Plotting "{}"...'.format(image_name))

    filepath = path.join(csv_dir, 'u_avgt.csv')
    if not path.exists(filepath):
        print("Plotting '{}' requires '{}'".format(image_name, filepath))
        return
    u_avgt = np.loadtxt(filepath, delimiter=',')

    filepath = path.join(csv_dir, 'v_avgt.csv')
    if not path.exists(filepath):
        print("Plotting '{}' requires '{}'".format(image_name, filepath))
        return
    v_avgt = np.loadtxt(filepath, delimiter=',')

    filepath = path.join(csv_dir, 'w_avgt.csv')
    if not path.exists(filepath):
        print("Plotting '{}' requires '{}'".format(image_name, filepath))
        return
    w_avgt = np.loadtxt(filepath, delimiter=',')

    plots_shape = np.array((2, 2))
    plots_size_each = np.array((6, 3))
    fig = plt.figure(figsize=plots_shape[::-1] * plots_size_each)

    ax = fig.add_subplot(*plots_shape, 1)
    ax.set_title(r"Time averaged $u$" if LATEX else "Time averaged u")
    pcm = ax.pcolormesh(x, z, u_avgt.T, shading='nearest', cmap="CMRmap", rasterized=True)
    fig.colorbar(pcm, ax=ax)
    ax.set_xlabel(r'$x$' if LATEX else 'x')
    ax.set_ylabel(r'$z$' if LATEX else 'z')

    ax = fig.add_subplot(*plots_shape, 2)
    ax.set_title(r"Time averaged $v$" if LATEX else "Time averaged v")
    pcm = ax.pcolormesh(x, z, v_avgt.T, shading='nearest', cmap="CMRmap", rasterized=True)
    fig.colorbar(pcm, ax=ax)
    ax.set_xlabel(r'$x$' if LATEX else 'x')
    ax.set_ylabel(r'$z$' if LATEX else 'z')

    ax = fig.add_subplot(*plots_shape, 3)
    ax.set_title(r"Time averaged $w$" if LATEX else "Time averaged w")
    pcm = ax.pcolormesh(x, z, w_avgt.T, shading='nearest', cmap="CMRmap", rasterized=True)
    fig.colorbar(pcm, ax=ax)
    ax.set_xlabel(r'$x$' if LATEX else 'x')
    ax.set_ylabel(r'$z$' if LATEX else 'z')

    fig.set_tight_layout(True)
    plt.savefig(path.join(plot_dir, image_name))
    plt.close()


def plot_temperature(csv_dir, plot_dir, x, y, z):
    image_name = "temperature.{}".format(FILE_EXT)
    print('Plotting "{}"...'.format(image_name))

    filepath = path.join(csv_dir, 'Ttop.csv')
    if not path.exists(filepath):
        print("Plotting '{}' requires '{}'".format(image_name, filepath))
        return
    Ttop = np.loadtxt(filepath, delimiter=',')

    filepath = path.join(csv_dir, 'Tmid.csv')
    if not path.exists(filepath):
        print("Plotting '{}' requires '{}'".format(image_name, filepath))
        return
    Tmid = np.loadtxt(filepath, delimiter=',')

    plots_shape = np.array((1, 2))
    plots_size_each = np.array((4, 4))
    fig = plt.figure(figsize=plots_shape[::-1] * plots_size_each)

    ax = fig.add_subplot(*plots_shape, 1)
    ax.set_title("Temperature top view")
    pcm = ax.pcolormesh(x, y, Ttop.T, shading='nearest', cmap="CMRmap", rasterized=True)
    fig.colorbar(pcm, ax=ax)
    ax.set_xlabel(r'$x$' if LATEX else 'x')
    ax.set_ylabel(r'$y$' if LATEX else 'y')
    ax.set_aspect(1)

    ax = fig.add_subplot(*plots_shape, 2)
    ax.set_title("Temperature side view")
    pcm = ax.pcolormesh(x, z, Tmid.T, shading='nearest', cmap="CMRmap", rasterized=True)
    fig.colorbar(pcm, ax=ax)
    ax.set_xlabel(r'$x$' if LATEX else 'x')
    ax.set_ylabel(r'$z$' if LATEX else 'z')
    ax.set_aspect(1)

    fig.set_tight_layout(True)
    plt.savefig(path.join(plot_dir, image_name))
    plt.close()


def plot_heat_flux_z(csv_dir, plot_dir, z):
    image_name = "heat_flux_z.{}".format(FILE_EXT)
    print('Plotting "{}"...'.format(image_name))

    filepath = path.join(csv_dir, 'heat_flux.csv')
    if not path.exists(filepath):
        print("Plotting '{}' requires '{}'".format(image_name, filepath))
        return

    data = np.loadtxt(filepath, delimiter=',')
    fluxconv = data[:, 0]
    fluxcond = data[:, 1]
    fluxtotal = fluxconv + fluxcond

    plots_shape = np.array((1, 1))
    plots_size_each = np.array((3.2, 3.2))
    fig = plt.figure(figsize=plots_shape[::-1] * plots_size_each)

    ax = fig.add_subplot(*plots_shape, 1)
    ax.set_title("Vertical heat flux")
    ax.plot(fluxcond, z, label="Conductive")
    ax.plot(fluxconv, z, label="Convective")
    ax.plot(fluxtotal, z, label="Total")
    ax.legend()
    ax.set_ylabel(r'$z$' if LATEX else 'z')

    fig.set_tight_layout(True)
    plt.savefig(path.join(plot_dir, image_name))
    plt.close()


def plot_energy(csv_dir, plot_dir, t):
    image_name = "energy.{}".format(FILE_EXT)
    print('Plotting "{}"...'.format(image_name))
    filepath = path.join(csv_dir, 'kinetic.csv')
    if not path.exists(filepath):
        print("Plotting '{}' requires '{}'".format(image_name, filepath))
        return
    KE = np.loadtxt(filepath, delimiter=',')

    plots_shape = np.array((1, 1))
    plots_size_each = np.array((4, 2.7))
    fig = plt.figure(figsize=plots_shape[::-1] * plots_size_each)

    ax = fig.add_subplot(*plots_shape, 1)
    ax.plot(t, KE)
    ax.set_ylabel('Kinetic Energy')
    ax.set_xlabel(r'$t$' if LATEX else 't')

    fig.set_tight_layout(True)
    plt.savefig(path.join(plot_dir, image_name))
    plt.close()


def plot_velocity_filters(csv_dir, plot_dir, x, z):
    """Plot a snapshot in time of the velocities, against the low and high pass filtered versions"""
    image_name = "velocity_filters.{}".format(FILE_EXT)
    print('Plotting "{}"...'.format(image_name))

    filenames = ("u_snapshot", "v_snapshot", "w_snapshot",
        "u_snapshot_lowpass", "v_snapshot_lowpass", "w_snapshot_lowpass",
        "u_snapshot_highpass", "v_snapshot_highpass", "w_snapshot_highpass")
    filepaths = [ path.join(csv_dir, filename+'.csv') for filename in filenames]
    for filepath in filepaths:
        if not path.exists(filepath):
            print("Plotting '{}' requires '{}'".format(image_name, filepath))
            return
    fields = [ np.loadtxt(file, delimiter=',') for file in filepaths ]

    plots_shape = np.array((3, 3))
    plots_size_each = np.array((4, 2.5))
    fig = plt.figure(figsize=plots_shape * plots_size_each)

    for i, name, field in zip(range(len(filenames)), filenames, fields):
        ax = fig.add_subplot(*plots_shape, i + 1)
        ax.set_title(name.replace('_', ' '))
        pcm = ax.pcolormesh(x, z, field.T, shading='nearest', cmap="CMRmap", rasterized=True)
        ax.set_xlabel(r'$x$' if LATEX else 'x')
        ax.set_ylabel(r'$z$' if LATEX else 'z')

    fig.set_tight_layout(True)
    plt.savefig(path.join(plot_dir, image_name))
    plt.close()


def plot_momentum_terms(csv_dir, plot_dir, z):
    image_name = "momentum_terms.{}".format(FILE_EXT)
    print('Plotting "{}"...'.format(image_name))
    filepath = path.join(csv_dir, 'momentum_terms.csv')
    if not path.exists(filepath):
        print("Plotting '{}' requires '{}'".format(image_name, filepath))
        return

    data = np.loadtxt(filepath, delimiter=',')
    viscous_x, viscous_y, coriolis_x, coriolis_y, rs_x, rs_y = [ data[:, i] for i in range(data.shape[1]) ]

    # Plot everything on two plots
    plots_shape = np.array((2, 1))
    plots_size_each = np.array((3.2, 2.7))
    fig = plt.figure(figsize=plots_shape[::-1] * plots_size_each)

    ax = fig.add_subplot(*plots_shape, 1)
    ax.set_title("Zonal Mean Flow")
    ax.axvline(0, lw=1, c='lightgray')
    ax.plot(viscous_x, z, label="Viscous", c='green', ls='-.')
    ax.plot(coriolis_x, z, label="Mean flow", c='blue', lw=2)
    ax.plot(rs_x, z, label="Stress", c='red')
    ax.set_ylabel(r'$z$' if LATEX else 'z')

    ax = fig.add_subplot(*plots_shape, 2)
    ax.set_title("Meridional Mean Flow")
    ax.axvline(0, lw=1, c='lightgray')
    ax.plot(viscous_y, z, label="Viscous", c='green', ls='-.')
    ax.plot(coriolis_y, z, label="Mean flow", c='blue', lw=2)
    ax.plot(rs_y, z, label="Stress", c='red')
    ax.set_ylabel(r'$z$' if LATEX else 'z')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), shadow=False, ncol=3, columnspacing=1, frameon=False)

    fig.set_tight_layout(True)
    plt.savefig(path.join(plot_dir, image_name))
    plt.close()


def plot_momentum_terms_filtered(data_dir, plot_dir, z):
    image_name = "momentum_terms_filtered.{}".format(FILE_EXT)
    print('Plotting "{}"...'.format(image_name))

    filepath = path.join(csv_dir, 'momentum_terms_filtered.csv')
    if not path.exists(filepath):
        print("Plotting '{}' requires '{}'".format(image_name, filepath))
        return
    data = np.loadtxt(filepath, delimiter=',')
    (coriolis_x, coriolis_y,
        viscous_x_low, viscous_x_high, viscous_y_low, viscous_y_high,
        rs_x_low, rs_x_high, rs_y_low, rs_y_high) = [ data[:, i] for i in range(data.shape[1]) ]

    # Plot everything on two plots
    plots_shape = np.array((2, 1))
    plots_size_each = np.array((3.2, 2.7))

    fig = plt.figure(figsize=plots_shape[::-1] * plots_size_each)

    ax = fig.add_subplot(*plots_shape, 1)
    ax.set_title("Zonal Mean Flow")
    ax.axvline(0, lw=1, c='darkgray')
    ax.plot(coriolis_x, z, label="Mean flow", c='lightblue', lw=3)
    # ax.plot(viscous_x_low, z, lw=1, ls='--', c='green')
    # ax.plot(viscous_x_high, z, lw=1, ls=':', c='green')
    ax.plot(rs_x_low, z, lw=1, ls='--', c='red')
    ax.plot(rs_x_high, z, lw=1, ls=':', c='red')
    ax.set_ylabel(r'$z$' if LATEX else 'z')

    ax = fig.add_subplot(*plots_shape, 2)
    ax.set_title("Meridional Mean Flow")
    ax.axvline(0, lw=1, c='darkgray')
    ax.plot(coriolis_y, z, label="Mean flow", c='lightblue', lw=3)
    # ax.plot(viscous_y_low, z, label="Viscous (lowpass)", lw=1, ls='--', c='green')
    # ax.plot(viscous_y_high, z, label="Viscous (highpass)", lw=1, ls=':', c='green')
    ax.plot(rs_y_low, z, label="Stress (lowpass)", lw=1, ls='--', c='red')
    ax.plot(rs_y_high, z, label="Stress (highpass)", lw=1, ls=':', c='red')
    ax.set_ylabel(r'$z$' if LATEX else 'z')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), shadow=False, ncol=1, columnspacing=1, frameon=False)

    fig.set_tight_layout(True)
    plt.savefig(path.join(plot_dir, image_name))
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide one argument: The file path to the directory to read the analysis from.")
        exit(1)
    data_dir = sys.argv[1]
    csv_dir = path.join(data_dir, "postproc/")
    plot_dir = sys.argv[2] if len(sys.argv) >= 3 else path.join(data_dir, "plots/")

    try:
        mkdir(plot_dir)
    except FileExistsError:
        pass

    if LATEX:
        plt.style.use("science")
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 10

    axis_names = ('t_analysis', 't', 'x', 'y', 'z', 'z_fourier')
    axis_files = [ path.join(csv_dir, 'axis_{}.csv'.format(axis)) for axis in axis_names ]
    axis_files = [ file if path.exists(file) else None for file in axis_files ]
    (t_analysis, t, x, y, z, z_fourier) = [ np.loadtxt(file, delimiter=',').flatten() if file else None for file in axis_files ]

    axis_shapes = [ len(axis) if axis is not None else 0 for axis in (t_analysis, t, x, y, z, z_fourier) ]
    for name, shape in zip(axis_names, axis_shapes):
        if shape > 0: print("Have axis '{}', length {}".format(name, shape))

    plot_stresses(csv_dir, plot_dir, z)
    plot_velocities(csv_dir, plot_dir, x, z)
    plot_temperature(csv_dir, plot_dir, x, y, z)
    plot_heat_flux_z(csv_dir, plot_dir, z)
    plot_energy(csv_dir, plot_dir, t_analysis)
    plot_velocity_filters(csv_dir, plot_dir, x, z_fourier)
    plot_momentum_terms(csv_dir, plot_dir, z)
    plot_momentum_terms_filtered(csv_dir, plot_dir, z_fourier)

    print("Done.")
