#
# This file has a bunch of stuff I've used previously, and I haven't
# used a lot of it in a while so zero guarantee that it works
#

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import Video
from numpy import fft
import os.path as path

import utils

def colorplot_matrix(params, x, y, m, labelx='x', labely='y', labelcolor='', title='', limits=(None, None)):
    fig = plt.figure(figsize=(5, 4), dpi=100)
    plt.pcolormesh(x, y, m.T, shading='nearest', cmap="coolwarm", vmin=limits[0], vmax=limits[1])
    plt.colorbar(label=labelcolor)
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.title(title, fontsize=9)
    plt.tight_layout()
    return fig

def find_peaks(arr):
    """Returns a filter for arr that selects only the values which are larger than both adjacent values"""
    arr_left = np.append(arr[1:], np.inf)
    arr_right = np.append([np.inf], arr[:-1])
    return (arr > arr_left) & (arr > arr_right)

def plot_power_spectra(params, quantity, t, x, z, name):
    
    avg_time_interval = 0.05 # Number of viscous times to average over
    
    # First average over time
    quantity2_avgt = np.mean((quantity * quantity)[np.where(t >= params['duration'] - avg_time_interval)], axis=0)
    # Then average over one of the spacial dimensions
    quantity2_avgtx = np.mean(quantity2_avgt, axis=0)
    quantity2_avgtz = np.mean(quantity2_avgt, axis=1)
    
    # Carry out fourier transform, then divide by wavenumbers to get power spectrum
    kx = 2 * np.pi * fft.fftfreq(len(x), d=params['Lx']/params['resX'])
    kz = 2 * np.pi * fft.fftfreq(len(z), d=params['Lz']/params['resZ'])
    power_z = np.abs(fft.fft(quantity2_avgtx, len(kz))) / kz
    power_x = np.abs(fft.fft(quantity2_avgtz, len(kx))) / kx
    
    # Remove the negative wavenumbers, and the zero frequency term
    kx_filter = kx > 0
    kz_filter = kz > 0
    kx = kx[kx_filter]
    kz = kz[kz_filter]
    power_z = power_z[kz_filter]
    power_x = power_x[kx_filter]
    
    # Find the peaks of the power spectrum so we can label them in the plot
    kx_peaks_filter = find_peaks(power_x) & (power_x > np.max(power_x) * 0.1)
    kz_peaks_filter = find_peaks(power_z) & (power_z > np.max(power_z) * 0.1)
    kx_peaks = kx[kx_peaks_filter]
    kz_peaks = kz[kz_peaks_filter]
    power_x_peaks = power_x[kx_peaks_filter]
    power_z_peaks = power_z[kz_peaks_filter]
    
    
    # Plot all the spectra
    fig = plt.figure(figsize=(8, 8), dpi=100)
    
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(kx, power_x)    
    for (x, y) in zip(kx_peaks, power_x_peaks):
        ax.scatter(x, y, marker='x', c='red')
        ax.annotate(' kx={0:.2f} '.format(x), (x, y))
    ax.set_xlabel('kx')
    ax.set_ylabel('P_x')
    
    ax = fig.add_subplot(2, 2, 2)
    ax.plot(kz, power_z)
    for (x, y) in zip(kz_peaks, power_z_peaks):
        ax.scatter(x, y, marker='x', c='red')
        ax.annotate(' kz={0:.2f} '.format(x), (x, y))
    ax.set_xlabel('kz')
    ax.set_ylabel('P_z')
    
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(2*np.pi/kx, power_x)
    for (x, y) in zip(2*np.pi/kx_peaks, power_x_peaks):
        ax.scatter(x, y, marker='x', c='red')
        ax.annotate(' lambda={0:.2f} '.format(x), (x, y))
    ax.set_xlabel('lambda x')
    ax.set_ylabel('P_x')
    
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(2*np.pi/kz, power_z)
    for (x, y) in zip(2*np.pi/kz_peaks, power_z_peaks):
        ax.scatter(x, y, marker='x', c='red')
        ax.annotate(' lambda={0:.2f} '.format(x), (x, y))
    ax.set_xlabel('lambda z')
    ax.set_ylabel('P_z')
    
    fig.suptitle(f'Power spectrum of {name} averaged over {avg_time_interval} viscous times\nP_x = F(<flux^2>_t,z) / kx')
    fig.tight_layout()


def average_temp(data_dir):
    params = utils.read_params(data_dir)
    with h5py.File("{}/analysis.h5".format(data_dir), mode='r') as file:
    
        # Load datasets
        task = file['tasks']['T']
        T = np.array(task)
        temp_avgx = np.mean(T, axis=1)
        t = task.dims[0]['sim_time']
        z = task.dims[2][0]

        # Average temp against t
        colorplot_matrix(params, t, z, temp_avgx, labelx='t', labely='z', labelcolor='<T>_x')
        plt.savefig("{}/temperature.png".format(data_dir))

def average_vel(data_dir):
    params = utils.read_params(data_dir)
    with h5py.File("{}/analysis.h5".format(data_dir), mode='r') as file:

        # Load datasets
        temp = file['tasks']['T']
        temp_avgz = np.squeeze(np.array(file['tasks']['<T>']))
        vel_avgz = np.squeeze(np.array(file['tasks']['<Re>']))
        t = temp.dims[0]['sim_time']
        z = temp.dims[2][0]

        # Average vel against t
        colorplot_matrix(params, t, z, vel_avgz, labelx='t', labely='z', labelcolor='<Re>(z)')
        plt.savefig("{}/velocity.png".format(data_dir))

def heat_flux(data_dir):
    return NotImplemented
    params = utils.read_params(data_dir)
    with h5py.File("{}/analysis.h5".format(data_dir), mode='r') as file:

        # Load datasets
        temp = file['tasks']['T']
        fluxconv = np.squeeze(np.array(file['tasks']['FluxConv']))
        fluxcond = np.squeeze(np.array(file['tasks']['FluxCond']))
        fluxtotal = fluxconv + fluxcond
        t = temp.dims[0]['sim_time']
        x = temp.dims[1][0]
        z = temp.dims[2][0]

        timerange = 0.1
        fluxconv_trim = fluxconv[np.where(t >= t[-1] - timerange)[0][0]:]
        fluxconv_avgt = np.average(fluxconv_trim, 0)
        fluxcond_trim = fluxcond[np.where(t >= t[-1] - timerange)[0][0]:]
        fluxcond_avgt = np.average(fluxcond_trim, 0)

        fluxtotal_avgt = fluxconv_avgt + fluxcond_avgt

        params_string = utils.create_params_string(params)
        fig = plt.figure(figsize=(7, 6), dpi=100)
        plt.plot(fluxconv_avgt, np.array(z), label="Convective")
        plt.plot(fluxcond_avgt, np.array(z), label="Conductive")
        plt.plot(fluxtotal_avgt, np.array(z), label="Total")
        plt.legend()
        plt.xlabel('Flux')
        plt.ylabel('z')
        #     plt.xlim(-1, np.amax(fluxconv_avgzt) + 1)
        plt.xlim(-1, 5)
        plt.title("Averaged over t={:.2f} to t={:.2f}\n{}".format(t[-1] - timerange, t[-1], params_string), fontsize=9)
        plt.tight_layout()
        plt.savefig("{}/heat_flux.png".format(data_dir))

def ang_mom_flux(data_dir):
    return NotImplemented
    params = utils.read_params(data_dir)
    with h5py.File("{}/analysis.h5".format(data_dir), mode='r') as file:

        # Load datasets
        timerange = 0.1
        temp = file['tasks']['T']
        t = np.array(temp.dims[0]['sim_time'])
        z = np.array(temp.dims[2][0])
        angmom = np.array(file['tasks']['AngMom'])
        u = np.array(file['tasks']['u'])
        flux = u * angmom
        flux_trim = flux[np.where(t >= t[-1] - timerange)[0][0]:]
        flux_avgt = np.average(flux_trim, 0)

        params_string = utils.create_params_string(params)
        fig = plt.figure(figsize=(5, 4), dpi=100)
        plt.plot(flux_avgt, z)
        plt.xlabel("Angular momentum flux (in x dir.)")
        plt.ylabel("z")
        plt.title("Averaged over t={:.2f} to t={:.2f}\n{}".format(t[-1] - timerange, t[-1], params_string), fontsize=9)
        plt.tight_layout()
        plt.savefig("{}/ang_mom_flux_avg.png".format(data_dir))

        colorplot_matrix(params, t, z, flux, labelx="t", labely="z", labelcolor="Angular momentum flux")
        plt.savefig("{}/ang_mom_flux.png".format(data_dir))

def video(data_dir):
    params = utils.read_params(data_dir)
    with h5py.File("{}/analysis.h5".format(data_dir), mode='r') as file:
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
        animation.save("{}/video.mp4".format(data_dir))
        plt.close()

def time_averaged(data_dir):
    params = utils.read_params(data_dir)
    with h5py.File("{}/analysis.h5".format(data_dir), mode='r') as file:
        avg_time_interval = 0.05 # Number of viscous times to average over
        
        t = np.array(file['tasks']['T'].dims[0]['sim_time'])
        x = np.array(file['tasks']['T'].dims[1][0])
        z = np.array(file['tasks']['T'].dims[2][0])
        
        num_plots = 4
        (figwidth, figheight) = utils.calc_plot_size(params)
        fig = plt.figure(figsize=(figwidth, figheight * num_plots), dpi=100)
        def plot(quantity, name, subplot):
            quantity_avgt = np.mean(quantity[np.where(t >= params['duration'] - avg_time_interval)], axis=0)
            ax = fig.add_subplot(num_plots, 1, subplot)
            # outer = np.amax(np.abs(quantity))
            pcm = ax.pcolormesh(x, z, quantity_avgt.T, shading='nearest', cmap="coolwarm")#, vmin=-outer, vmax=outer)
            fig.colorbar(pcm, label=name, ax=ax)
            ax.set_xlabel('x')
            ax.set_ylabel('z')
        
        # Velocity
        w = np.array(file['tasks']['w'])
        u = np.array(file['tasks']['u'])
        vel = np.sqrt(u*u + w*w)
        plot(vel, "Velocity", 1)

        # Temperature
        T = np.array(file['tasks']['T'])
        plot(T, "Temperature", 2)
        
        # Heat flux
        T = np.array(file['tasks']['T'])
        w = np.array(file['tasks']['w'])
        Tz = np.array(file['tasks']['Tz'])
        heatflux = T * w - Tz
        plot(heatflux, "Heat Flux", 3)
        
        # uw (Reynolds stress)
        uw = u * w
        plot(uw, "uw (Reynolds stress)", 4)
        
#         # Ang mom
#         angmom = np.array(file['tasks']['AngMom'])
#         plot(angmom, "Angular Momentum", 4)
        
#         # Ang mom flux
#         angmomflux = np.array(file['tasks']['v']) * angmom
#         plot(angmomflux, "Angular Momentum Flux", 5)
        
#         # x velocity
#         u = np.array(file['tasks']['u'])
#         plot(u, "x velocity", 6)
        
#         # y velocity
#         v = np.array(file['tasks']['v'])
#         plot(v, "y velocity", 7)
        
#         # z velocity
#         w = np.array(file['tasks']['w'])
#         plot(w, "z velocity", 8)
        
        fig.suptitle(f'Quantities time-averaged from {params["duration"] - avg_time_interval} to {params["duration"]} viscous times')
        fig.tight_layout()
        plt.savefig(path.join(data_dir, "time_averaged.jpg"))

def final_state(data_dir):
    params = utils.read_params(data_dir)
    with h5py.File("{}/analysis.h5".format(data_dir), mode='r') as file:
        avg_time_interval = 0.05 # Number of viscous times to average over
        
        t = np.array(file['tasks']['T'].dims[0]['sim_time'])
        x = np.array(file['tasks']['T'].dims[1][0])
        z = np.array(file['tasks']['T'].dims[2][0])
        
        num_plots = 5
        (figwidth, figheight) = utils.calc_plot_size(params)
        fig = plt.figure(figsize=(figwidth, figheight * num_plots), dpi=100)
        def plot(quantity, name, subplot):
            quantity_final = quantity[-1]
            ax = fig.add_subplot(num_plots, 1, subplot)
            pcm = ax.pcolormesh(x, z, quantity_final.T, shading='nearest', cmap="coolwarm")
            fig.colorbar(pcm, label=name, ax=ax)
            ax.set_xlabel('x')
            ax.set_ylabel('z')
        
        # Velocity
        w = np.array(file['tasks']['w'])
        u = np.array(file['tasks']['u'])
        vel = np.sqrt(u*u + w*w)
        plot(vel, "Velocity", 1)

        # Temperature
        T = np.array(file['tasks']['T'])
        plot(T, "Temperature", 2)
        
        # Heat flux
        T = np.array(file['tasks']['T'])
        w = np.array(file['tasks']['w'])
        Tz = np.array(file['tasks']['Tz'])
        heatflux = T * w - Tz
        plot(heatflux, "Heat Flux", 3)
        
        # Ang mom
        angmom = np.array(file['tasks']['AngMom'])
        plot(angmom, "Angular Momentum", 4)
        
        # Ang mom flux
        angmomflux = np.array(file['tasks']['v']) * angmom
        plot(angmomflux, "Angular Momentum Flux", 5)
        
        fig.suptitle(f'Quantities at the end of the simulation ({params["duration"]} viscous times)')
        fig.tight_layout()
        plt.savefig(path.join(data_dir, "final_state.jpg"))

# def final_state(data_dir):
#     params = utils.read_params(data_dir)
#     with h5py.File("{}/analysis.h5".format(data_dir), mode='r') as file:

#         # Load datasets
#         temp = file['tasks']['T']
#         t = temp.dims[0]['sim_time']
#         x = temp.dims[1][0]
#         z = temp.dims[2][0]

#         params_string = utils.create_params_string(params)
#         fig = plt.figure(figsize=utils.calc_plot_size(params), dpi=100)
#         quad = plt.pcolormesh(x, z, temp[-1].T, shading='nearest', cmap="coolwarm")
#         plt.colorbar()
#         def animate(frame):
#             # For some reason, matplotlib ends up getting the x and y axes the wrong way round,
#             # so I just took the transpose of each frame to 'fix' it.
#             quad.set_array(frame.T)
#         plt.xlabel('x')
#         plt.ylabel('z')
#         plt.title("Final state (t={:.2f})\n{}".format(params["duration"], params_string), fontsize=9)
#         plt.savefig("{}/final_state.png".format(data_dir))

def angmom_flux_spectra(data_dir):
    params = utils.read_params(data_dir)
    with h5py.File("{}/analysis.h5".format(data_dir), mode='r') as file:
        task = file['tasks']['AngMom']
        angmom = np.array(task)
        flux = np.array(file['tasks']['v']) * angmom
        t = np.array(task.dims[0]['sim_time'])
        x = np.array(task.dims[1][0])
        z = np.array(task.dims[2][0])
        plot_power_spectra(params, flux, t, x, z, "angular momentum flux (into the page)")
        plt.savefig(path.join(data_dir, "angmom_flux_spectra.jpg"))

def heat_flux_spectra(data_dir):
    params = utils.read_params(data_dir)
    with h5py.File("{}/analysis.h5".format(data_dir), mode='r') as file:
        T = np.array(file['tasks']['T'])
        w = np.array(file['tasks']['w'])
        Tz = np.array(file['tasks']['Tz'])
        flux = T * w - Tz
        t = np.array(file['tasks']['T'].dims[0]['sim_time'])
        x = np.array(file['tasks']['T'].dims[1][0])
        z = np.array(file['tasks']['T'].dims[2][0])
        plot_power_spectra(params, flux, t, x, z, "vertical heat flux")
        plt.savefig(path.join(data_dir, "heat_flux_spectra.jpg"))
