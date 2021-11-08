#!/usr/bin/env python

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import Video
from numpy import fft
import os.path as path
import sys

import utils

def plot_filter_comparison(data_dir):
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

        fig = plt.figure(figsize=np.flip(plots_shape) * plots_size_each)
        fig.suptitle(f'Averaged over {np.round(params["average_interval"], 2)} viscous times')

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
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please provide one argument: The file path to the directory to read the analysis from.")
        exit(1)
    data_dir = sys.argv[1]

    plot_filter_comparison(data_dir)
