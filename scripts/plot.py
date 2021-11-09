#!/usr/bin/env python

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import Video
from numpy import fft
from os import path
from os import mkdir
import sys

import utils

def plot_velocities(data_dir, image_name):
    print(f'Plotting "{image_name}"...')
    params = utils.read_params(data_dir)
    with h5py.File(path.join(data_dir, 'averaged', 'averaged.h5'), mode='r') as file:

        t = np.array(file['tasks']['u'].dims[0]['sim_time'])
        x = np.array(file['tasks']['u'].dims[1][0])
        z = np.array(file['tasks']['u'].dims[2][0])

        u = np.array(file['tasks']['u'])[-1]
        v = np.array(file['tasks']['v'])[-1]
        w = np.array(file['tasks']['w'])[-1]

        plots_shape = np.array((2, 2))
        plots_size_each = np.array((8, 4))

        tstart = 0 if len(t) == 1 else t[-2]
        tend = t[-1]
        fig = plt.figure(figsize=np.flip(plots_shape) * plots_size_each)
        fig.suptitle(f'Averaged from {np.round(tstart, 2)} to {np.round(tend, 2)} viscous times')

        ax = fig.add_subplot(*plots_shape, 1)
        ax.set_title("Time averaged u")
        pcm = ax.pcolormesh(x, z, u.T, shading='nearest', cmap="CMRmap", label='<u>')
        fig.colorbar(pcm, ax=ax)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_aspect(1)

        ax = fig.add_subplot(*plots_shape, 2)
        ax.set_title("Time averaged v")
        pcm = ax.pcolormesh(x, z, v.T, shading='nearest', cmap="CMRmap", label='<v>')
        fig.colorbar(pcm, ax=ax)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_aspect(1)

        ax = fig.add_subplot(*plots_shape, 3)
        ax.set_title("Time averaged w")
        pcm = ax.pcolormesh(x, z, w.T, shading='nearest', cmap="CMRmap", label='<w>')
        fig.colorbar(pcm, ax=ax)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_aspect(1)

        plt.tight_layout()
        plt.savefig(path.join(data_dir, 'plots', image_name))
        plt.close()

def plot_velocities_post(data_dir, image_name):
    print(f'Plotting "{image_name}"...')
    params = utils.read_params(data_dir)
    with h5py.File(path.join(data_dir, 'averaged', 'averaged.h5'), mode='r') as file:

        t = np.array(file['tasks']['u'].dims[0]['sim_time'])
        x = np.array(file['tasks']['u'].dims[1][0])
        z = np.array(file['tasks']['u'].dims[2][0])

        u = np.array(file['tasks']['u'])[t >= params['duration'] - params['average_interval']]
        v = np.array(file['tasks']['v'])[t >= params['duration'] - params['average_interval']]
        w = np.array(file['tasks']['w'])[t >= params['duration'] - params['average_interval']]

        u_avgt = np.mean(u, axis=0)
        v_avgt = np.mean(v, axis=0)
        w_avgt = np.mean(w, axis=0)

        plots_shape = np.array((2, 2))
        plots_size_each = np.array((8, 4))

        tstart = params['duration'] - params['average_interval']
        tend = params['duration']
        fig = plt.figure(figsize=np.flip(plots_shape) * plots_size_each)
        fig.suptitle(f'Averaged from {np.round(tstart, 2)} to {np.round(tend, 2)} viscous times')

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
        plt.savefig(path.join(data_dir, 'plots', image_name))
        plt.close()

def plot_stresses(data_dir, image_name):
    print(f'Plotting "{image_name}"...')
    params = utils.read_params(data_dir)
    with h5py.File(path.join(data_dir, 'averaged', 'averaged.h5'), mode='r') as file:

        t = np.array(file['tasks']['u'].dims[0]['sim_time'])
        x = np.array(file['tasks']['u'].dims[1][0])
        z = np.array(file['tasks']['u'].dims[2][0])

        stress_uw = np.array(file['tasks']['stress_uw'])[-1]
        stress_vw = np.array(file['tasks']['stress_vw'])[-1]
        stress_uw_dz = np.squeeze(np.array(file['tasks']['stress_uw_avgx_dz'])[-1])
        stress_vw_dz = np.squeeze(np.array(file['tasks']['stress_vw_avgx_dz'])[-1])

        plots_shape = np.array((2, 2))
        plots_size_each = np.array((8, 4))

        tstart = 0 if len(t) == 1 else t[-2]
        tend = t[-1]
        fig = plt.figure(figsize=np.flip(plots_shape) * plots_size_each)
        fig.suptitle(f'Averaged from {np.round(tstart, 2)} to {np.round(tend, 2)} viscous times')

        ax = fig.add_subplot(*plots_shape, 1)
        ax.set_title("stress_uw")
        pcm = ax.pcolormesh(x, z, stress_uw.T, shading='nearest', cmap="CMRmap")
        fig.colorbar(pcm, ax=ax)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_aspect(1)

        ax = fig.add_subplot(*plots_shape, 2)
        ax.set_title("stress_vw")
        pcm = ax.pcolormesh(x, z, stress_vw.T, shading='nearest', cmap="CMRmap")
        fig.colorbar(pcm, ax=ax)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_aspect(1)

        ax = fig.add_subplot(*plots_shape, 3)
        ax.set_title("dz(stress)")
        ax.plot(stress_uw_dz, z, label='dz(<uw>)')
        ax.plot(stress_vw_dz, z, label='dz(<vw>)')
        ax.set_xlabel('dz(stress)')
        ax.set_ylabel('z')
        ax.legend()

        plt.tight_layout()
        plt.savefig(path.join(data_dir, 'plots', image_name))
        plt.close()

def plot_stresses_post(data_dir, image_name):
    print(f'Plotting "{image_name}"...')
    params = utils.read_params(data_dir)
    with h5py.File(path.join(data_dir, 'analysis.h5'), mode='r') as file:

        t = np.array(file['tasks']['u'].dims[0]['sim_time'])
        x = np.array(file['tasks']['u'].dims[1][0])
        z = np.array(file['tasks']['u'].dims[2][0])

        u = np.array(file['tasks']['u'])[t >= params['duration'] - params['average_interval']]
        v = np.array(file['tasks']['v'])[t >= params['duration'] - params['average_interval']]
        w = np.array(file['tasks']['w'])[t >= params['duration'] - params['average_interval']]
        u_avgt = np.mean(u, axis=0)
        v_avgt = np.mean(v, axis=0)
        w_avgt = np.mean(w, axis=0)
        u_pert = u - u_avgt
        v_pert = v - v_avgt
        w_pert = w - w_avgt
        stress_uw_avgt = np.mean(u_pert*w_pert, axis=0)
        stress_vw_avgt = np.mean(v_pert*w_pert, axis=0)
        stress_uw_avgt_dz = np.diff(np.mean(stress_uw_avgt, axis=0), n=1, axis=-1)
        stress_vw_avgt_dz = np.diff(np.mean(stress_vw_avgt, axis=0), n=1, axis=-1)

        plots_shape = np.array((2, 2))
        plots_size_each = np.array((8, 4))

        tstart = params['duration'] - params['average_interval']
        tend = params['duration']
        fig = plt.figure(figsize=np.flip(plots_shape) * plots_size_each)
        fig.suptitle(f'Averaged from {np.round(tstart, 2)} to {np.round(tend, 2)} viscous times')

        ax = fig.add_subplot(*plots_shape, 1)
        ax.set_title("stress_uw")
        pcm = ax.pcolormesh(x, z, stress_uw_avgt.T, shading='nearest', cmap="CMRmap")
        fig.colorbar(pcm, ax=ax)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_aspect(1)

        ax = fig.add_subplot(*plots_shape, 2)
        ax.set_title("stress_vw")
        pcm = ax.pcolormesh(x, z, stress_vw_avgt.T, shading='nearest', cmap="CMRmap")
        fig.colorbar(pcm, ax=ax)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_aspect(1)

        ax = fig.add_subplot(*plots_shape, 3)
        ax.set_title("dz(stress)")
        ax.plot(stress_uw_avgt_dz, z[:-1], label='dz(<uw>)')
        ax.plot(stress_vw_avgt_dz, z[:-1], label='dz(<vw>)')
        ax.set_xlabel('dz(stress)')
        ax.set_ylabel('z')
        ax.legend()

        plt.tight_layout()
        plt.savefig(path.join(data_dir, 'plots', image_name))
        plt.close()

def plot_filter_comparison(data_dir, image_name):
    print(f'Plotting "{image_name}"...')
    params = utils.read_params(data_dir)
    with h5py.File(path.join(data_dir, 'averaged', 'averaged.h5'), mode='r') as file:

        t = np.array(file['tasks']['u'].dims[0]['sim_time'])
        x = np.array(file['tasks']['u'].dims[1][0])
        z = np.array(file['tasks']['u'].dims[2][0])


        # Plot the stresses in 2d

        stress_uw = np.array(file['tasks']['stress_uw'])[-1]
        stress_uw_low = np.array(file['tasks']['stress_uw_low'])[-1]
        stress_uw_high = np.array(file['tasks']['stress_uw_high'])[-1]

        plots_shape = np.array((2, 2))
        plots_size_each = np.array((8, 4))

        tstart = 0 if len(t) == 1 else t[-2]
        tend = t[-1]
        fig = plt.figure(figsize=np.flip(plots_shape) * plots_size_each)
        fig.suptitle(f'Averaged from {np.round(tstart, 2)} to {np.round(tend, 2)} viscous times')

        ax = fig.add_subplot(*plots_shape, 1)
        ax.set_title("stress_uw")
        pcm = ax.pcolormesh(x, z, stress_uw.T, shading='nearest', cmap="CMRmap")
        fig.colorbar(pcm, ax=ax)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_aspect(1)

        ax = fig.add_subplot(*plots_shape, 2)
        ax.set_title("stress_uw lowpass filtered")
        pcm = ax.pcolormesh(x, z, stress_uw_low.T, shading='nearest', cmap="CMRmap")
        fig.colorbar(pcm, ax=ax)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_aspect(1)

        ax = fig.add_subplot(*plots_shape, 3)
        ax.set_title("stress_uw highpass filtered")
        pcm = ax.pcolormesh(x, z, stress_uw_high.T, shading='nearest', cmap="CMRmap")
        fig.colorbar(pcm, ax=ax)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_aspect(1)

        # Plot the z differential of the stresses

        stress_uw_dz = np.squeeze(np.array(file['tasks']['stress_uw_avgx_dz'])[-1])
        stress_uw_low_dz = np.squeeze(np.array(file['tasks']['stress_uw_low_avgx_dz'])[-1])
        stress_uw_high_dz = np.squeeze(np.array(file['tasks']['stress_uw_high_avgx_dz'])[-1])

        ax = fig.add_subplot(*plots_shape, 4)
        ax.set_title("stress_uw_dz")
        ax.plot(stress_uw_dz, z, label='no filter')
        ax.plot(stress_uw_low_dz, z, label='lowpass')
        ax.plot(stress_uw_high_dz, z, label='highpass')
        ax.set_xlabel('stress_uw_dz')
        ax.set_ylabel('z')
        ax.legend()

        plt.tight_layout()
        plt.savefig(path.join(data_dir, 'plots', image_name))
        plt.close()

def video(data_dir):
    print(f'Rendering video...')
    params = utils.read_params(data_dir)
    with h5py.File(path.join(data_dir, 'analysis.h5'), mode='r') as file:
        # Load datasets
        temp = file['tasks']['T']
        t = temp.dims[0]['sim_time']
        x = temp.dims[1][0]
        z = temp.dims[2][0]

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
        s_per_visc_time = 30
        animation = ani.FuncAnimation(fig, animate, frames=temp, interval=params['timestep_analysis']*s_per_visc_time*1000)
        animation.save(path.join(data_dir, 'plots', 'video.mp4'))
        plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please provide one argument: The file path to the directory to read the analysis from.")
        exit(1)
    data_dir = sys.argv[1]

    try:
        mkdir(path.join(data_dir, 'plots'))
    except FileExistsError:
        pass

    plot_velocities(data_dir, 'velocities.jpg')
    plot_velocities_post(data_dir, 'velocities_post.jpg')
    plot_stresses(data_dir, 'stresses.jpg')
    plot_stresses_post(data_dir, 'stresses_post.jpg')
    plot_filter_comparison(data_dir, 'filter_comparison_stresses.jpg')
    # video(data_dir)
