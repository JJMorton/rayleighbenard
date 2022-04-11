#!/usr/bin/env python3

# This script calculates all the things we want to plot and
# saves them to csv files. You can run this script on isca
# and then transfer all the small csv files to your computer
# and plot them there

import h5py
import numpy as np
from os import path
from os import mkdir
import sys

import utils

from memory_profiler import memory_usage

def get_field(file, fieldname):
    """Get a field from the .h5 file, with dimensions (t, x, y, z)"""
    return file['tasks'][fieldname]

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
    """Average the field along the zonal direction: (x, y, z) --> (x, z)"""
    arr = np.array(arr)
    dims = len(arr.shape)
    if dims != 3:
        raise Exception("Attempt to zonally average array with {} dimensions".format(dims))
    return np.mean(arr, axis=1)

def average_horizontal(arr):
    """Average the field horizonally: (x, y, z) --> (z,)"""
    arr = np.array(arr)
    dims = len(arr.shape)
    if dims != 3:
        raise Exception("Attempt to horizontally average array with {} dimensions".format(dims))
    return np.mean(np.mean(arr, axis=0), axis=0)

# Numpy gradient function along the z direction, working with numpy ver >= 1.9.1
def gradientz(arr, z):
    spacings = [np.arange(res) for res in arr.shape][:-1]
    grad = np.gradient(arr, *spacings, z, edge_order=2)
    if type(grad) == list:
        return grad[-1]
    else:
        return grad

def average_in_time(t, time_start, func):
    """
    Apply the function `func` to each timestep of the task provided,
    and then average in time, starting from the provided index.
    This function is needed, because we want to avoid bringing the
    entire task into memory, as over all the timesteps it can be
    enormous. Make sure to avoid converting any large fields into
    a numpy array (or splicing them) before passing them through this
    function.
    """
    time_indices = np.arange(len(t))[t >= time_start]
    out_shape = func(time_indices[0]).shape
    field_average = np.zeros(out_shape)
    for i in time_indices:
        field_average += func(i)
    return field_average / len(time_indices)

def momentum_terms(params, t, z, u, v, w):
    """
    Returns the terms of the averaged momentum equation, with signs as they are
    in the form "mean flow = RS + viscous", i.e. the mean flow should be equal
    to the sum of the other two terms.
    """

    duration = min(params['duration'], t[-1])
    if duration < params['average_interval']: print('WARNING: averaging interval longer than simulation duration, averaging over entire duration...')
    tstart = max(duration - params['average_interval'], 0)

    # We have cos here (instead of sin as in other research) because we are taking
    # theta as an angle from the vertical, not from the latitude
    coeff = np.cos(params["Theta"]) * params["Ta"]**0.5

    print("    Viscous X")
    viscous_x = average_horizontal(average_in_time(t, tstart, lambda i: -gradientz(gradientz(u[i], z), z) / coeff))
    print("    Viscous Y")
    viscous_y = average_horizontal(average_in_time(t, tstart, lambda i: gradientz(gradientz(v[i], z), z) / coeff))
    # viscous_y = gradientz(gradientz(v, z), z) / coeff
    # viscous_y_avg = np.mean(average_horizontal(viscous_y), axis=0)
    # del viscous_y

    print("    Coriolis X")
    coriolis_x = average_in_time(t, tstart, lambda i: average_horizontal(v[i]))
    # coriolis_x_avg = np.mean(average_horizontal(v), axis=0)
    print("    Coriolis Y")
    coriolis_y = average_in_time(t, tstart, lambda i: average_horizontal(u[i]))

    print("    Time averages")
    u_avgt = np.array(average_in_time(t, tstart, lambda i: u[i]))
    v_avgt = np.array(average_in_time(t, tstart, lambda i: v[i]))
    w_avgt = np.array(average_in_time(t, tstart, lambda i: w[i]))

    print("    RS X")
    # stress_x = np.mean(gradientz(average_horizontal((u - u_avgt) * (w - w_avgt)), z), axis=0) / coeff
    stress_x = average_in_time(t, tstart, lambda i: (np.array(u[i]) - u_avgt) * (np.array(w[i]) - w_avgt)) / coeff
    stress_x = gradientz(average_horizontal(stress_x), z)
    print("    RS Y")
    # stress_y = -np.mean(gradientz(average_horizontal((v - v_avgt) * (w - w_avgt)), z), axis=0) / coeff
    stress_y = -average_in_time(t, tstart, lambda i: (np.array(v[i]) - v_avgt) * (np.array(w[i]) - w_avgt)) / coeff
    stress_y = gradientz(average_horizontal(stress_y), z)

    return viscous_x, viscous_y, coriolis_x, coriolis_y, stress_x, stress_y


def calc_velocities(data_dir, output_dir):
    """Plot the time averaged velocities"""
    print('Calculating average velocities')
    params = utils.read_params(data_dir)
    filepath = path.join(data_dir, 'vel.h5')
    if not path.exists(filepath):
        print("  '{}' is required, skipping".format(filepath))
        return

    with h5py.File(filepath, mode='r') as file:

        t, x, y, z = get_dims(file, 'u')

        duration = min(params['duration'], t[-1])
        if duration < params['average_interval']: print('WARNING: averaging interval longer than simulation duration, averaging over entire duration...')
        tstart = max(duration - params['average_interval'], 0)

        u = get_field(file, 'u')
        u_avgt = average_in_time(t, tstart, lambda i: average_zonal(u[i]))
        output_file = path.join(output_dir, "u_avgt.csv")
        np.savetxt(output_file, u_avgt, delimiter=',', header="u_avgt(x, z)")

        v = get_field(file, 'v')
        v_avgt = average_in_time(t, tstart, lambda i: average_zonal(v[i]))
        output_file = path.join(output_dir, "v_avgt.csv")
        np.savetxt(output_file, v_avgt, delimiter=',', header="v_avgt(x, z)")

        w = get_field(file, 'w')
        w_avgt = average_in_time(t, tstart, lambda i: average_zonal(w[i]))
        output_file = path.join(output_dir, "w_avgt.csv")
        np.savetxt(output_file, w_avgt, delimiter=',', header="w_avgt(x, z)")


def calc_temperature(data_dir, output_dir):
    print('Calculating temperature snapshots...')
    filepath = path.join(data_dir, 'analysis.h5')
    if not path.exists(filepath):
        print("  '{}' is required, skipping".format(filepath))
        return
    with h5py.File(filepath, mode='r') as file:
        Ttop = np.squeeze(get_field(file, 'Ttop')[-1])
        Tmid = np.squeeze(get_field(file, 'Tmid')[-1])
        output_file = path.join(output_dir, "Ttop.csv")
        np.savetxt(output_file, Ttop, delimiter=',', header="Ttop(x, y)")
        output_file = path.join(output_dir, "Tmid.csv")
        np.savetxt(output_file, Tmid, delimiter=',', header="Tmid(x, z)")


def calc_heat_flux_z(data_dir, output_dir):
    print('Calculating heat fluxes...')
    params = utils.read_params(data_dir)
    filepath = path.join(data_dir, 'analysis.h5')
    if not path.exists(filepath):
        print("  '{}' is required, skipping".format(filepath))
        return

    with h5py.File(filepath, mode='r') as file:

        t = get_dims(file, 'FluxHeatConv')[0]

        duration = min(params['duration'], t[-1])
        if duration < params['average_interval']: print('WARNING: averaging interval longer than simulation duration, averaging over entire duration...')
        tstart = max(duration - params['average_interval'], 0)

        task = get_field(file, 'FluxHeatConv')
        fluxconv = average_in_time(t, tstart, func=lambda i: average_horizontal(task[i]))
        task = get_field(file, 'FluxHeatCond')
        fluxcond = average_in_time(t, tstart, func=lambda i: average_horizontal(task[i]))

        output_file = path.join(output_dir, "heat_flux.csv")
        np.savetxt(
            output_file,
            np.stack((fluxconv.T, fluxcond.T), axis=1),
            delimiter=',',
            header="fluxconv(z),fluxcond(z)"
        )


def calc_energy(data_dir, output_dir):
    print('Calculating kinetic energy...')
    filepath = path.join(data_dir, 'analysis.h5')
    if not path.exists(filepath):
        print("  '{}' is required, skipping".format(filepath))
        return

    with h5py.File(filepath, mode='r') as file:
        KE = np.squeeze(get_field(file, 'E'))
        output_file = path.join(output_dir, "kinetic.csv")
        np.savetxt(output_file, KE.T, delimiter=',', header="KE(t)")


def calc_velocity_filters(data_dir, output_dir):
    """Plot a snapshot in time of the velocities, against the low and high pass filtered versions"""
    print('Calculating velocity filter snapshots...')
    params = utils.read_params(data_dir)
    filepath1 = path.join(data_dir, 'interp_u.h5')
    filepath2 = path.join(data_dir, 'interp_v.h5')
    filepath3 = path.join(data_dir, 'interp_w.h5')
    if not path.exists(filepath1):
        print("  '{}' is required, skipping".format(filepath1))
        return
    if not path.exists(filepath2):
        print("  '{}' is required, skipping".format(filepath2))
        return
    if not path.exists(filepath3):
        print("  '{}' is required, skipping".format(filepath3))
        return

    with h5py.File(filepath1, mode='r') as file:

        t = get_dims_interp(file)[0]

        duration = min(params['duration'], t[-1])
        if duration < params['average_interval']: print('WARNING: averaging interval longer than simulation duration, averaging over entire duration...')
        tstart = duration - params['average_interval']

        # Correct the order of the axes after reading in the fields
        u = get_field(file, 'u')[-1][:, params["resY"]//2]
        u_lowpass = get_field(file, 'u_lowpass')[-1][:, params["resY"]//2]
        u_highpass = get_field(file, 'u_highpass')[-1][:, params["resY"]//2]

    with h5py.File(filepath2, mode='r') as file:
        v = get_field(file, 'v')[-1][:, params["resY"]//2]
        v_lowpass = get_field(file, 'v_lowpass')[-1][:, params["resY"]//2]
        v_highpass = get_field(file, 'v_highpass')[-1][:, params["resY"]//2]
    
    with h5py.File(filepath3, mode='r') as file:
        w = get_field(file, 'w')[-1][:, params["resY"]//2]
        w_lowpass = get_field(file, 'w_lowpass')[-1][:, params["resY"]//2]
        w_highpass = get_field(file, 'w_highpass')[-1][:, params["resY"]//2]

    names = ("u_snapshot", "v_snapshot", "w_snapshot",
        "u_snapshot_lowpass", "v_snapshot_lowpass", "w_snapshot_lowpass",
        "u_snapshot_highpass", "v_snapshot_highpass", "w_snapshot_highpass")
    fields = (u, v, w, u_lowpass, v_lowpass, w_lowpass, u_highpass, v_highpass, w_highpass)
    for name, field in zip(names, fields):
        output_file = path.join(output_dir, name + ".csv")
        np.savetxt(output_file, field, delimiter=',', header="{}(x, z_fourier)".format(name))


def calc_stresses(data_dir, output_dir):
    print('Calculating Reynolds stresses')
    params = utils.read_params(data_dir)
    filepath = path.join(data_dir, 'vel.h5')
    if not path.exists(filepath):
        print("  '{}' is required, skipping".format(filepath))
        return

    print("  Reading velocity fields from file...")
    with h5py.File(filepath, mode='r') as file:

        t = get_dims(file, 'u')[0]

        duration = min(params['duration'], t[-1])
        if duration < params['average_interval']: print('WARNING: averaging interval longer than simulation duration, averaging over entire duration...')
        tstart = max(duration - params['average_interval'], 0)

        u = get_field(file, 'u')
        v = get_field(file, 'v')
        w = get_field(file, 'w')

        print("  Calculating stresses...")
        u_avgt = np.array(average_in_time(t, tstart, lambda i: u[i]))
        v_avgt = np.array(average_in_time(t, tstart, lambda i: v[i]))
        w_avgt = np.array(average_in_time(t, tstart, lambda i: w[i]))
        stress_uw = average_in_time(t, tstart, lambda i: average_horizontal((np.array(u[i]) - u_avgt) * (np.array(w[i]) - w_avgt)))
        stress_vw = average_in_time(t, tstart, lambda i: average_horizontal((np.array(v[i]) - v_avgt) * (np.array(w[i]) - w_avgt)))
        stress_uv = average_in_time(t, tstart, lambda i: average_horizontal((np.array(u[i]) - u_avgt) * (np.array(v[i]) - v_avgt)))

    output_file = path.join(output_dir, "stresses.csv")
    np.savetxt(
        output_file,
        np.stack((stress_uw.T, stress_vw.T, stress_uv.T), axis=1),
        delimiter=',',
        header="stress_uw(z),stress_vw(z),stress_uv(z)"
    )


def calc_momentum_terms(data_dir, output_dir):
    print('Calculating averaged momentum terms...')
    params = utils.read_params(data_dir)
    filepath = path.join(data_dir, 'vel.h5')
    if not path.exists(filepath):
        print("  '{}' is required, skipping".format(filepath))
        return

    print("  Reading velocity fields from file...")
    with h5py.File(filepath, mode='r') as file:

        t, x, y, z = get_dims(file, 'u')

        u = get_field(file, 'u')
        v = get_field(file, 'v')
        w = get_field(file, 'w')

        print("  Calculating terms...")
        viscous_x, viscous_y, coriolis_x, coriolis_y, rs_x, rs_y = momentum_terms(params, t, z, u, v, w)

    output_file = path.join(output_dir, "momentum_terms.csv")
    np.savetxt(
        output_file,
        np.stack((viscous_x.T, viscous_y.T, coriolis_x.T, coriolis_y.T, rs_x.T, rs_y.T), axis=1),
        delimiter=',',
        header="viscous_x(z),viscous_y(z),coriolis_x(z),coriolis_y(z),rs_x(z),rs_y(z)"
    )


def calc_momentum_terms_filtered(data_dir, output_dir):
    print('Calculating filtered averaged momentum terms...')
    params = utils.read_params(data_dir)
    filepath1 = path.join(data_dir, 'interp_u.h5')
    filepath2 = path.join(data_dir, 'interp_v.h5')
    filepath3 = path.join(data_dir, 'interp_w.h5')
    if not path.exists(filepath1):
        print("  '{}' is required, skipping".format(filepath1))
        return
    if not path.exists(filepath2):
        print("  '{}' is required, skipping".format(filepath2))
        return
    if not path.exists(filepath3):
        print("  '{}' is required, skipping".format(filepath3))
        return

    print("  Reading unfiltered fields from files...")
    with h5py.File(filepath1, mode='r') as file:
        t, x, y, z = get_dims_interp(file)
        u = get_field(file, 'u')
        coriolis_y = average_in_time(t, 0, lambda i: average_horizontal(u[i]))
    with h5py.File(filepath2, mode='r') as file:
        v = get_field(file, 'v')
        coriolis_x = average_in_time(t, 0, lambda i: average_horizontal(v[i]))

    print("  Reading lowpassed fields from files...")
    with h5py.File(filepath1, mode='r') as file:
        u_low = get_field(file, 'u_lowpass')
        with h5py.File(filepath2, mode='r') as file:
            v_low = get_field(file, 'v_lowpass')
            with h5py.File(filepath3, mode='r') as file:
                w_low = get_field(file, 'w_lowpass')
                print("  Calculating lowpass filtered terms...")
                viscous_x_low, viscous_y_low, _, _, rs_x_low, rs_y_low = momentum_terms(params, t, z, u_low, v_low, w_low)

    print("  Reading highpassed fields from files...")
    with h5py.File(filepath1, mode='r') as file:
        u_high = get_field(file, 'u_highpass')
        with h5py.File(filepath2, mode='r') as file:
            v_high = get_field(file, 'v_highpass')
            with h5py.File(filepath3, mode='r') as file:
                w_high = get_field(file, 'w_highpass')
                print("  Calculating highpass filtered terms...")
                viscous_x_high, viscous_y_high, _, _, rs_x_high, rs_y_high = momentum_terms(params, t, z, u_high, v_high, w_high)

    output_file = path.join(output_dir, "momentum_terms_filtered.csv")
    np.savetxt(
        output_file,
        np.stack((coriolis_x.T, coriolis_y.T, viscous_x_low.T, viscous_x_high.T, viscous_y_low.T, viscous_y_high.T, rs_x_low.T, rs_x_high.T, rs_y_low.T, rs_y_high.T), axis=1),
        delimiter=',',
        header="coriolis_x(z_fourier),coriolis_y(z_fourier),viscous_x_low(z_fourier),viscous_x_high(z_fourier),viscous_y_low(z_fourier),viscous_y_high(z_fourier),rs_x_low(z_fourier),rs_x_high(z_fourier),rs_y_low(z_fourier),rs_y_high(z_fourier)"
    )


def save_axes(data_dir, output_dir):
    print("Saving axes...")

    filepath = path.join(data_dir, 'analysis.h5')
    if path.exists(filepath):
        with h5py.File(filepath, mode='r') as file:
            t_analysis = get_dims(file, 'E')[0]
        output_file = path.join(output_dir, "axis_t_analysis.csv")
        np.savetxt(output_file, t_analysis.T, delimiter=',')
        print("  Saved t_analysis")
    else:
        print("  Analysis file doesn't exist, skipping...")

    filepath = path.join(data_dir, 'vel.h5')
    # Use axes from analysis.h5 if vel.h5 doesn't exist
    if not path.exists(filepath):
        filepath = path.join(data_dir, 'analysis.h5')
    if path.exists(filepath):
        with h5py.File(filepath, mode='r') as file:
            t, x, y, z = get_dims(file, 'u')
        output_file = path.join(output_dir, "axis_t.csv")
        np.savetxt(output_file, t.T, delimiter=',')
        output_file = path.join(output_dir, "axis_x.csv")
        np.savetxt(output_file, x.T, delimiter=',')
        output_file = path.join(output_dir, "axis_y.csv")
        np.savetxt(output_file, y.T, delimiter=',')
        output_file = path.join(output_dir, "axis_z.csv")
        np.savetxt(output_file, z.T, delimiter=',')
        print("  Saved t, x, y and z")
    else:
        print("  '{}' doesn't exist, skipping...".format(filepath))

    filepath = path.join(data_dir, 'interp_u.h5')
    if path.exists(filepath):
        with h5py.File(filepath, mode='r') as file:
            t_interp, x, y, z_fourier = get_dims_interp(file)
        output_file = path.join(output_dir, "axis_t_interp.csv")
        np.savetxt(output_file, t_interp.T, delimiter=',')
        output_file = path.join(output_dir, "axis_z_fourier.csv")
        np.savetxt(output_file, z_fourier.T, delimiter=',')
        print("  Saved z_fourier")
    else:
        print("  Interpolated fields don't exist (use interpolate.py script to create them), skipping...".format(filepath))


def main():
    if len(sys.argv) < 2:
        print("Please provide one argument: The file path to the directory to read the analysis from.")
        exit(1)
    data_dir = sys.argv[1]
    output_dir = path.join(data_dir, "postproc/")

    try:
        mkdir(output_dir)
    except FileExistsError:
        pass

    save_axes(data_dir, output_dir)
    calc_velocities(data_dir, output_dir)
    calc_temperature(data_dir, output_dir)
    calc_heat_flux_z(data_dir, output_dir)
    calc_energy(data_dir, output_dir)
    calc_velocity_filters(data_dir, output_dir)
    calc_stresses(data_dir, output_dir)
    calc_momentum_terms(data_dir, output_dir)
    calc_momentum_terms_filtered(data_dir, output_dir)

    print("Done.")

if __name__ == "__main__":
    mem_usage = memory_usage(main)
    print('Maximum memory usage: %sMB' % max(mem_usage))

