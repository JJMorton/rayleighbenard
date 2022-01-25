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
    task = file['tasks'][fieldname]
    arr = np.array(task)
    num_dims = len(task.dims)
    if num_dims == 4:
        # This was a 3D simulation, average and flatten along zonal direction
        # dims are (t, x, y, z)
        arr = np.mean(arr, axis=2)
    return arr

def get_dims(file, fieldname):
    task = file['tasks'][fieldname]
    num_dims = len(task.dims)
    dims = (
        np.array(task.dims[0]['sim_time']),
        np.array(task.dims[1][0]),
        np.array(task.dims[num_dims - 1][0])
    )
    return dims # (t, x, z)


def plot_velocities(data_dir, plot_dir, image_name):
    print(f'Plotting "{image_name}"...')
    params = utils.read_params(data_dir)
    with h5py.File(path.join(data_dir, 'analysis.h5'), mode='r') as file:

        t, x, z = get_dims(file, 'u')

        duration = min(params['duration'], t[-1])
        if duration < params['average_interval']: print('WARNING: averaging interval longer than simulation duration, averaging over entire duration...')
        timeframe_mask = np.logical_and(t >= duration - params['average_interval'], t <= duration)

        t = t[timeframe_mask]

        u = get_field(file, 'u')[timeframe_mask]
        v = get_field(file, 'v')[timeframe_mask]
        w = get_field(file, 'w')[timeframe_mask]

        u_avgt = np.mean(u, axis=0)
        v_avgt = np.mean(v, axis=0)
        w_avgt = np.mean(w, axis=0)

        plots_shape = np.array((2, 2))
        plots_size_each = np.array((8, 4))

        tstart = duration - params['average_interval']
        tend = duration
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
        plt.savefig(path.join(plot_dir, image_name))
        plt.close()


def plot_stresses(data_dir, plot_dir, image_name):
    print(f'Plotting "{image_name}"...')
    params = utils.read_params(data_dir)
    with h5py.File(path.join(data_dir, 'analysis.h5'), mode='r') as file:

        t, x, z = get_dims(file, 'u')

        duration = min(params['duration'], t[-1])
        if duration < params['average_interval']: print('WARNING: averaging interval longer than simulation duration, averaging over entire duration...')
        timeframe_mask = np.logical_and(t >= duration - params['average_interval'], t <= duration)

        t = t[timeframe_mask]

        u = get_field(file, 'u')[timeframe_mask]
        v = get_field(file, 'v')[timeframe_mask]
        w = get_field(file, 'w')[timeframe_mask]
        u_avgt = np.mean(u, axis=0, keepdims=True)
        v_avgt = np.mean(v, axis=0, keepdims=True)
        w_avgt = np.mean(w, axis=0, keepdims=True)
        u_pert = u - u_avgt
        v_pert = v - v_avgt
        w_pert = w - w_avgt
        stress_uw_avgt = np.mean(u_pert*w_pert, axis=0)
        stress_vw_avgt = np.mean(v_pert*w_pert, axis=0)
        stress_uw_avgt_dz = np.gradient(np.mean(stress_uw_avgt, axis=0), z, axis=-1, edge_order=2)
        stress_vw_avgt_dz = np.gradient(np.mean(stress_vw_avgt, axis=0), z, axis=-1, edge_order=2)

        plots_shape = np.array((2, 2))
        plots_size_each = np.array((8, 4))

        tstart = duration - params['average_interval']
        tend = duration
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
        ax.set_title("horizontally averaged stresses")
        ax.plot(np.mean(stress_uw_avgt, axis=0), z, label='<uw>')
        ax.plot(np.mean(stress_vw_avgt, axis=0), z, label='<vw>')
        ax.set_xlabel('stress')
        ax.set_ylabel('z')
        ax.legend()

        ax = fig.add_subplot(*plots_shape, 4)
        ax.set_title("dz(horizontally averaged stress)")
        ax.plot(stress_uw_avgt_dz, z, label='dz(<uw>)')
        ax.plot(stress_vw_avgt_dz, z, label='dz(<vw>)')
        ax.set_xlabel('dz(stress)')
        ax.set_ylabel('z')
        ax.legend()

        plt.tight_layout()
        plt.savefig(path.join(plot_dir, image_name))
        plt.close()


def plot_heat_flux_z(data_dir, plot_dir, image_name):
    print(f'Plotting "{image_name}"...')
    params = utils.read_params(data_dir)
    with h5py.File(path.join(data_dir, 'analysis.h5'), mode='r') as file:

        t, x, z = get_dims(file, 'u')

        duration = min(params['duration'], t[-1])
        if duration < params['average_interval']: print('WARNING: averaging interval longer than simulation duration, averaging over entire duration...')
        timeframe_mask = np.logical_and(t >= duration - params['average_interval'], t <= duration)

        t = t[timeframe_mask]

        T = get_field(file, 'T')[timeframe_mask]
        Tz = get_field(file, 'Tz')[timeframe_mask]
        w = get_field(file, 'w')[timeframe_mask]

        fluxconv = np.mean(np.mean(T * w, axis=1), axis=0)
        fluxcond = np.mean(np.mean(-Tz, axis=1), axis=0)
        fluxtotal = fluxconv + fluxcond

        plots_shape = np.array((1, 1))
        plots_size_each = np.array((8, 4))

        tstart = duration - params['average_interval']
        tend = duration
        fig = plt.figure(figsize=np.flip(plots_shape) * plots_size_each)
        fig.suptitle(f'Averaged from {np.round(tstart, 2)} to {np.round(tend, 2)} viscous times')

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


def plot_pressure_profile(data_dir, plot_dir, image_name):
    print(f'Plotting "{image_name}"...')
    params = utils.read_params(data_dir)
    with h5py.File(path.join(data_dir, 'analysis.h5'), mode='r') as file:
        t, x, z = get_dims(file, 'u')

        duration = min(params['duration'], t[-1])
        if duration < params['average_interval']: print('WARNING: averaging interval longer than simulation duration, averaging over entire duration...')
        timeframe_mask = np.logical_and(t >= duration - params['average_interval'], t <= duration)

        t = t[timeframe_mask]

        p = get_field(file, 'p')[timeframe_mask]
        p_avg = np.mean(np.mean(p, axis=0), axis=0)

        plots_shape = np.array((1, 1))
        plots_size_each = np.array((8, 4))

        tstart = duration - params['average_interval']
        tend = duration
        fig = plt.figure(figsize=np.flip(plots_shape) * plots_size_each)
        fig.suptitle(f'Averaged from {np.round(tstart, 2)} to {np.round(tend, 2)} viscous times')

        ax = fig.add_subplot(*plots_shape, 1)
        ax.set_title("Time- and horizonally-averaged pressure")
        ax.plot(p_avg, z)
        ax.set_xlabel('<P>')
        ax.set_ylabel('z')

        plt.tight_layout()
        plt.savefig(path.join(plot_dir, image_name))
        plt.close()


def plot_energy(data_dir, plot_dir, image_name):
    print(f'Plotting "{image_name}"...')
    params = utils.read_params(data_dir)
    with h5py.File(path.join(data_dir, 'analysis.h5'), mode='r') as file:
        t, x, z = get_dims(file, 'u')

        u = get_field(file, 'u')
        v = get_field(file, 'v')
        w = get_field(file, 'w')

        KE = np.sum(0.5 * (u*u + v*v + w*w), axis=(1, 2))

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


def plot_momentum(data_dir, plot_dir, image_name):
    print(f'Plotting "{image_name}"...')
    params = utils.read_params(data_dir)
    with h5py.File(path.join(data_dir, 'analysis.h5'), mode='r') as file:
        t, x, z = get_dims(file, 'u')

        u = get_field(file, 'u')[timeframe_mask]
        v = get_field(file, 'v')[timeframe_mask]
        w = get_field(file, 'w')[timeframe_mask]

        px = np.sum(u, axis=(1, 2))
        py = np.sum(v, axis=(1, 2))
        pz = np.sum(w, axis=(1, 2))
        p = np.sum(u + v + w, axis=(1, 2))

        plots_shape = np.array((1, 1))
        plots_size_each = np.array((8, 4))

        fig = plt.figure(figsize=np.flip(plots_shape) * plots_size_each)

        ax = fig.add_subplot(*plots_shape, 1)
        ax.set_title("Total momentum as a function of time")
        ax.plot(t, px, label='x-component', lw=1, c='red')
        ax.plot(t, py, label='y-component', lw=1, c='green')
        ax.plot(t, pz, label='z-component', lw=1, c='blue')
        ax.plot(t, p, label='total', ls='--', lw=1, c='black')
        ax.set_ylabel('Momentum')
        ax.set_xlabel('t')
        ax.legend()

        plt.tight_layout()
        plt.savefig(path.join(plot_dir, image_name))
        plt.close()


def plot_momentum_x_terms(data_dir, plot_dir, image_name):
    print(f'Plotting "{image_name}"...')
    params = utils.read_params(data_dir)
    with h5py.File(path.join(data_dir, 'analysis.h5'), mode='r') as file:

        t, x, z = get_dims(file, 'u')

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
        viscous_np = -np.gradient(np.gradient(u, z, axis=2, edge_order=2), z, axis=2, edge_order=2)
        inertial = w * u_dz + u * u_dx
        inertial_np = w * np.gradient(u, z, axis=2, edge_order=2) + u * np.gradient(u, x, axis=1, edge_order=2)
        coriolis = v * np.sin(params["Theta"]) * params["Ta"]**0.5

        snapshot_time_index = -1
        snapshot_time = t[snapshot_time_index]
        temporal_avgx = np.mean(temporal[snapshot_time_index], axis=0)
        temporal_np_avgx = np.mean(temporal_np[snapshot_time_index], axis=0)
        viscous_avgx = np.mean(viscous[snapshot_time_index], axis=0)
        viscous_np_avgx = np.mean(viscous_np[snapshot_time_index], axis=0)
        inertial_avgx = np.mean(inertial[snapshot_time_index], axis=0)
        inertial_np_avgx = np.mean(inertial_np[snapshot_time_index], axis=0)
        coriolis_avgx = np.mean(coriolis[snapshot_time_index], axis=0)
        total_np_avgx = temporal_np_avgx + viscous_np_avgx + inertial_np_avgx - coriolis_avgx
        total_avgx = temporal_avgx + viscous_avgx + inertial_avgx - coriolis_avgx

        stress_avgxt = np.mean(np.gradient(np.mean( (u - np.mean(u, keepdims=True)) * (w - np.mean(w, keepdims=True)) , axis=0), z, axis=1, edge_order=2), axis=0)
        temporal_avgxt = np.mean(np.mean(temporal, axis=0), axis=0)
        temporal_np_avgxt = np.mean(np.mean(temporal_np, axis=0), axis=0)
        viscous_avgxt = np.mean(np.mean(viscous, axis=0), axis=0)
        viscous_np_avgxt = np.mean(np.mean(viscous_np, axis=0), axis=0)
        coriolis_avgxt = np.mean(np.mean(coriolis, axis=0), axis=0)
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


def plot_momentum_y_terms(data_dir, plot_dir, image_name):
    print(f'Plotting "{image_name}"...')
    params = utils.read_params(data_dir)
    with h5py.File(path.join(data_dir, 'analysis.h5'), mode='r') as file:

        t, x, z = get_dims(file, 'u')

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
        viscous_np = np.gradient(np.gradient(v, z, axis=2, edge_order=2), z, axis=2, edge_order=2)
        inertial = -(w * v_dz + u * v_dx)
        inertial_np = -(w * np.gradient(v, z, axis=2, edge_order=2) + u * np.gradient(v, x, axis=1, edge_order=2))
        # coriolis = (u * np.sin(params["Theta"]) - w * np.cos(params['Theta'])) / params["Ek"]
        coriolis = u * np.sin(params["Theta"]) * params["Ta"]**0.5

        snapshot_time_index = -1
        snapshot_time = t[snapshot_time_index]
        temporal_avgx = np.mean(temporal[snapshot_time_index], axis=0)
        temporal_np_avgx = np.mean(temporal_np[snapshot_time_index], axis=0)
        viscous_avgx = np.mean(viscous[snapshot_time_index], axis=0)
        viscous_np_avgx = np.mean(viscous_np[snapshot_time_index], axis=0)
        inertial_avgx = np.mean(inertial[snapshot_time_index], axis=0)
        inertial_np_avgx = np.mean(inertial_np[snapshot_time_index], axis=0)
        coriolis_avgx = np.mean(coriolis[snapshot_time_index], axis=0)
        total_np_avgx = temporal_np_avgx + viscous_np_avgx + inertial_np_avgx - coriolis_avgx
        total_avgx = temporal_avgx + viscous_avgx + inertial_avgx - coriolis_avgx

        stress_avgxt = -np.mean(np.gradient(np.mean( (v - np.mean(v, keepdims=True)) * (w - np.mean(w, keepdims=True)) , axis=0), z, axis=1, edge_order=2), axis=0)
        temporal_avgxt = np.mean(np.mean(temporal, axis=1), axis=0)
        temporal_np_avgxt = np.mean(np.mean(temporal_np, axis=1), axis=0)
        viscous_avgxt = np.mean(np.mean(viscous, axis=1), axis=0)
        viscous_np_avgxt = np.mean(np.mean(viscous_np, axis=0), axis=0)
        coriolis_avgxt = np.mean(np.mean(coriolis, axis=0), axis=0)
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


def plot_temperature(data_dir, plot_dir, image_name):
    print(f'Plotting "{image_name}"...')
    params = utils.read_params(data_dir)
    with h5py.File(path.join(data_dir, 'analysis.h5'), mode='r') as file:

        t, x, z = get_dims(file, 'u')

        duration = min(params['duration'], t[-1])
        if duration < params['average_interval']: print('WARNING: averaging interval longer than simulation duration, averaging over entire duration...')
        timeframe_mask = np.logical_and(t >= duration - params['average_interval'], t <= duration)

        t = t[timeframe_mask]

        T = get_field(file, 'T')[timeframe_mask]
        T_avgt = np.mean(T, axis=0)

        plots_shape = np.array((1, 1))
        plots_size_each = np.array((8, 4))

        tstart = duration - params['average_interval']
        tend = duration
        fig = plt.figure(figsize=np.flip(plots_shape) * plots_size_each)
        fig.suptitle(f'Averaged from {np.round(tstart, 2)} to {np.round(tend, 2)} viscous times')

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


def video(data_dir, plot_dir):
    print(f'Rendering video...')
    params = utils.read_params(data_dir)
    with h5py.File(path.join(data_dir, 'analysis.h5'), mode='r') as file:
        # Load datasets
        temp = get_field(file, 'T')[timeframe_mask]
        t, x, z = get_dims(file, 'T')

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

    # plot_pressure_profile(data_dir, plot_dir, 'pressure.jpg')
    plot_velocities(data_dir, plot_dir, 'velocities.jpg')
    plot_stresses(data_dir, plot_dir, 'stresses.jpg')
    plot_momentum_x_terms(data_dir, plot_dir, 'momentum_x_terms.jpg')
    plot_momentum_y_terms(data_dir, plot_dir, 'momentum_y_terms.jpg')
    plot_heat_flux_z(data_dir, plot_dir, 'heat_flux_z.jpg')
    plot_energy(data_dir, plot_dir, 'energy.jpg')
    # plot_momentum(data_dir, plot_dir, 'momentum.jpg')
    plot_temperature(data_dir, plot_dir, 'temperature.jpg')
    # video(data_dir, plot_dir)
