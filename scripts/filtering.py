#!/usr/bin/env python3

import numpy as np
from dedalus import public as de
from matplotlib import pyplot as plt
import numpy.fft as fft
import time
try:
    from perlin_noise import PerlinNoise
except ImportError:
    PerlinNoise = None

from scipy import signal

def create_linear_bases(bases):
    """
    Converts any non evenly-spaced bases into evenly-spaced ones.
    Bases supplied should be monotonically increasing.
    """
    newbases = []
    for basis in bases:
        # Create a linear basis to interpolate to
        basis_lin = np.linspace(np.amin(basis), np.amax(basis), len(basis))
        newbases.append(basis_lin)
    return newbases

# This function is no longer used, only in test cases below
def interp_to_basis(arr, axis=-1, src=de.Chebyshev, dest=de.Fourier, dest_res=None):
    """
    Interpolate `arr` into an evenly spaced grid along `axis` (the last
    axis by default)
    """

    # Put the axis we are interpolating first
    arr = np.swapaxes(arr, 0, axis)
    num_dims = len(arr.shape)

    dest_res = dest_res or arr.shape[0]

    # Construct the dedalus domain and field to be interpolated
    # It doesn't matter what basis we use for the axes that aren't being interpolated, they are left as is
    bases = [ de.Fourier('b{}'.format(i), res, interval=(0, 1)) for i, res in zip(range(num_dims), arr.shape) ]
    bases[0] = src('z', arr.shape[0], interval=(0, 1))
    domain = de.Domain(bases)
    field = domain.new_field()
    field['g'] = arr

    # The axis to be interpolated onto
    shape_dest = list(arr.shape)
    shape_dest[0] = dest_res
    interped = np.zeros(shape_dest)
    z_grid_dest = dest('z', dest_res, interval=(0, 1)).grid()

    # Interpolation
    for i in range(len(z_grid_dest)):
        interp = de.operators.interpolate(field, z=z_grid_dest[i]).evaluate()
        if interp is None:
            raise Exception("Failed to interpolate field {}".format(field.name))
        interped[i] = np.real(interp['g'])[0]

    # Return the axes to their original order
    return np.swapaxes(interped, 0, axis)


def fft_nd(data, axes, bases):
    """
    Takes the N-dimensional fourier transform of the array `data` along the specified axes.
    `axes` is a list of indices specifying the dimensions of `data` to transform.
    `bases` is a list of the bases (i.e. what you would plot `data` against, like x, y and z).
    """

    if len(axes) != len(bases):
        raise Exception("fft_nd: Number of axes must equal number of bases")

    def basis_has_uniform_spacing(basis):
        epsilon = 1e-15
        # Calculates array of differences between adjacent elements, subtracts the difference between the first
        # two elements, and tests if these are all basically zero
        return np.all(np.abs((basis[1:] - basis[:-1]) - (basis[1] - basis[0])) < epsilon)

    # Make sure the axes are uniformly spaced all the way along
    for basis in bases:
        if not basis_has_uniform_spacing(basis):
            raise Exception("Attempt to apply an FFT to an axis with non-uniform spacing")

    # Sampling interval of each axis
    spacing = [ basis[1] - basis[0] for basis in bases ]

    # Calculate the wavenumbers to plot against in k-space
    # fftfreq returns frequencies, we want wavenumbers (i.e. 2*pi*f)
    kbases = [ 2 * np.pi * fft.fftshift(fft.fftfreq(len(basis), d=d)) for (basis, d) in zip(bases, spacing) ]

    # Actually perform the FFT
    transformed = fft.fftshift(fft.fftn(data, axes=axes))

    return transformed, kbases


def fft_nd_inverse(data, axes):
    """
    Takes the N-dimensional inverse fourier transform of the array `data` along the specified axes.
    `axes` is a list of indices specifying the dimensions of `data` to transform.
    """
    return np.real(fft.ifftn(fft.ifftshift(data), axes=axes))


def radius_filter(data_fft, kbases, kmin=0, kmax=np.inf):
    # k_space_magnitude[i, j] = kz[i]^2 + kx[j]^2
    # i.e. each element is the magnitude of the corresponding wave-vector
    k_space_magnitude = np.array(0)
    for basis in kbases: k_space_magnitude = np.add.outer(k_space_magnitude, basis*basis)

    # the masks are 2d arrays of booleans that can be used to select only low/high frequencies from a transformed array
    mask = np.logical_and(k_space_magnitude < kmax*kmax, k_space_magnitude >= kmin*kmin)

    # Apply the mask
    return np.where(mask, data_fft, 0)


def butterworth_filter(data_fft, kbases, kmin=0, kmax=np.inf):
    ftype = None
    omega = None
    if kmin > 0 and kmax < np.inf:
        ftype = "bandpass"
        omega = (kmin, kmax)
    elif kmin > 0:
        ftype = "highpass"
        omega = kmin
    elif kmax < np.inf:
        ftype = "lowpass"
        omega = kmax
    else:
        raise ValueError("Specify a finite value for either kmin or kmax")

    k_space_magnitude = np.array(0)
    for basis in kbases: k_space_magnitude = np.add.outer(k_space_magnitude, basis*basis)
    b, a = signal.butter(10, omega, ftype, analog=True)
    _, h = signal.freqs(b, a, worN=k_space_magnitude)
    return data_fft * np.abs(h)


def kspace_lowpass(data, axes, bases, lambda_cutoff):
    """
    Apply a lowpass filter to `data` along the specified axes.
    `bases` is the set of bases that data is specified by (e.g. x, y, and z arrays).
    `lambda_cutoff` is the smallest wavelength (equivalent to the largest frequency) that will be retained.

    Example usage: kspace_lowpass(3d_arr, (0, 1, 2), (x, y, z), Lz / 4)
        where x, y and z are 1-D arrays specifying the axes' grid points (i.e. the grid scales from Dedalus)
    """

    # Transform to k space
    data_fft, kbases = fft_nd(data, axes, bases)

    # Apply the mask
    # the masks are 2d arrays of booleans that can be used to select only low/high frequencies from a transformed array
    cutoff = 2 * np.pi / lambda_cutoff
    data_fft = butterworth_filter(data_fft, kbases, kmax=cutoff)

    # Invert the FFT back to real space
    data_fft_reversed = fft_nd_inverse(data_fft, axes)

    # Interpolate back to the original basis
    return data_fft_reversed


def kspace_highpass(data, axes, bases, lambda_cutoff):
    """
    Apply a highpass filter to `data` along the specified axes.
    `bases` is the set of bases that data is specified by (e.g. x, y, and z arrays).
    `lambda_cutoff` is the largest wavelength (equivalent to the smallest frequency) that will be retained.

    Example usage: kspace_highpass(3d_arr, (0, 1, 2), (x, y, z), Lz / 4)
        where x, y and z are 1-D arrays specifying the axes' grid points (i.e. the grid scales from Dedalus)
    """

    # Transform to k space
    data_fft, kbases = fft_nd(data, axes, bases)

    # Apply the mask
    # the masks are 2d arrays of booleans that can be used to select only low/high frequencies from a transformed array
    cutoff = 2 * np.pi / lambda_cutoff
    data_fft = butterworth_filter(data_fft, kbases, kmin=cutoff)

    # Invert the FFT back to real space
    data_fft_reversed = fft_nd_inverse(data_fft, axes)

    # Interpolate back to the original basis
    return data_fft_reversed


#======================================================
# FROM HERE ONWARDS,
# FUNCTIONS FOR TESTING THE INTERPOLATION AND FILTERING

def time_execution():

    def run_test(res, dims, interpolate):
        """
        Plots a test case of the filtering functions in this file
        """

        grid_shape = tuple([res] * dims)
        L = 1
        wavelength_cutoff = L / 2 # Radius in wavelength-space

        print("  Timing filtering of a {} array with{} interpolation...".format('x'.join([str(res)] * dims), "" if interpolate else "out"))

        zaxis = de.Chebyshev('z', res, interval=(0, L)).grid(scale=1)
        xaxis = de.Fourier('x', res, interval=(0, L)).grid(scale=1)
        axes = [xaxis] * dims
        axes[-1] = zaxis
        data = np.ones(grid_shape) * np.sin(4 * np.pi * zaxis)

        start_time = time.time()
        data_filter_low = kspace_lowpass(data, tuple(range(dims)), axes, wavelength_cutoff, interp=interpolate)
        duration_low = np.round(time.time() - start_time, 3)

        start_time = time.time()
        data_filter_high = kspace_highpass(data, tuple(range(dims)), axes, wavelength_cutoff, interp=interpolate)
        duration_high = np.round(time.time() - start_time, 3)

        return duration_low, duration_high

    gs = plt.GridSpec(1, 4)
    fig = plt.figure(figsize=(gs.ncols * 4, gs.nrows * 3))
    res = np.arange(5, 100, 5)
    res4d = np.arange(5, 55, 5)
    print("Running tests for 2 dimensions")
    dims2 = np.array([ np.mean(run_test(r, 2, interpolate=True)) for r in res ])
    print("Running tests for 3 dimensions")
    dims3 = np.array([ np.mean(run_test(r, 3, interpolate=True)) for r in res ])
    print("Running tests for 4 dimensions")
    dims4 = np.array([ np.mean(run_test(r, 4, interpolate=True)) for r in res4d ])

    print("  Calculating polyfit for 3D...")

    poly = np.polyfit(res, dims3, deg=3)
    print("  coeffs = {}".format(poly))
    polyx = np.linspace(0, 256, 256)
    polyy = poly[0] * polyx**3 + poly[1] * polyx**2 + poly[2] * polyx + poly[3]

    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(res, dims2, marker='x')
    ax.set_xlabel("Resolution")
    ax.set_ylabel("Time taken (s)")
    ax.set_title("2 dimensions")

    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(res, dims3, marker='x')
    ax.plot(polyx, polyy, label="Cubic fit")
    ax.axvline(polyx[-1], ls='--')
    ax.axhline(polyy[-1], ls='--', label=str(np.round(polyy[-1], 3)) + "s")
    ax.set_xlabel("Resolution")
    ax.set_ylabel("Time taken (s)")
    ax.legend()
    ax.set_title("3 dimensions")

    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(res4d, dims4, marker='x')
    ax.set_xlabel("Resolution")
    ax.set_ylabel("Time taken (s)")
    ax.set_title("4 dimensions")

    ax = fig.add_subplot(gs[0, 3])
    ax.plot(res, res, ls='--', color='lightgray')
    ax.scatter(res4d, dims4/dims3[:len(res4d)], label="4-D / 3-D", marker='x')
    ax.scatter(res, dims3/dims2, label="3-D / 2-D", marker='x')
    ax.legend()
    ax.set_xlabel("Resolution")
    ax.set_ylabel("Ratio of time taken")
    ax.set_title("Less than linear --> better\nin higher dimensions")

    fig.suptitle("Time taken to interpolate + filter arrays with different resolutions & dimensions")

    plt.tight_layout()
    plt.show()

def plot_test_case():
    """
    Plots a test case of the filtering functions in this file
    """

    print("Running filtering test case")

    print("  Creating data...")
    # Use non-equal lengths and resolutions to make sure everything works with this
    Lx = 1
    Lz = 1
    resX = 70
    resZ = 50
    wavelength_cutoff = Lx / 2 # Radius in wavelength-space

    x = np.linspace(0, Lx, resX)
    z_cheb = de.Chebyshev('z', resZ, interval=(0, Lz)).grid(scale=1)

    data = np.sin(4 * np.pi * np.add.outer(x, np.zeros(len(z_cheb)))) + np.sin(8 * np.pi * np.add.outer(np.zeros(len(x)), z_cheb))
    (x, z_lin) = create_linear_bases((x, z_cheb))
    data_interp = interp_to_basis(data, axis=-1, src=de.Chebyshev, dest=de.Fourier)

    print("  Calculating transforms...")
    data_fft, (kx, kz) = fft_nd(data, (0, 1), (x, z_lin))
    data_fft_abs = np.abs(data_fft)
    data_fft_inv = fft_nd_inverse(data_fft, (0, 1))
    data_filter_low = kspace_lowpass(data, (0, 1), (x, z_cheb), wavelength_cutoff, interp=False)
    data_filter_high = kspace_highpass(data, (0, 1), (x, z_cheb), wavelength_cutoff, interp=False)

    data_interp_fft, (kx, kz) = fft_nd(data_interp, (0, 1), (x, z_lin))
    data_interp_fft_abs = np.abs(data_interp_fft)
    data_interp_fft_inv = fft_nd_inverse(data_interp_fft, (0, 1))
    data_interp_filter_low = kspace_lowpass(data, (0, 1), (x, z_cheb), wavelength_cutoff, interp=True)
    data_interp_filter_high = kspace_highpass(data, (0, 1), (x, z_cheb), wavelength_cutoff, interp=True)

    print("  Plotting...")
    gs = plt.GridSpec(2, 5)
    fig = plt.figure(figsize=(gs.ncols * 4, gs.nrows * 3))
    fig.suptitle(
        "High- and low-pass filtering with and without interpolation.\n" +
        "The first row shows filtering the simple (but incorrect) way, simply doing the FFT using a linear basis instead of the Chebyshev one.\n" +
        "The second row is the same process, but the z-axis is interpolated into a linear basis first.\n" +
        "Note the noisy (and incorrect) frequencies for the z-axis in the first case, and how they're mostly fixed in the second."
    )

    ##################################
    # 1st row, original data, FFT of original data w.r.t. linear basis (clearly wrong approach)

    ax = fig.add_subplot(gs[0, 0])
    mesh = ax.pcolormesh(x, z_cheb, data.T, shading='nearest')
    plt.colorbar(mesh)
    ax.set_title("Original data: sin(8πz/Lz) + sin(2πx/Lx)")
    ax.set_xlabel("x")
    ax.set_ylabel("z (Chebyshev)")

    ax = fig.add_subplot(gs[0, 1])
    mesh = ax.pcolormesh(kx, kz, data_fft_abs.T, shading='nearest')
    plt.colorbar(mesh)
    ax.set_title("FFT (magnitude) using linear basis")
    ax.set_xlabel("kx")
    ax.set_ylabel("kz")

    ax = fig.add_subplot(gs[0, 2])
    mesh = ax.pcolormesh(x, z_cheb, (data_fft_inv - data).T, shading='nearest')
    plt.colorbar(mesh)
    ax.set_title("Inverse of FFT minus untransformed")
    ax.set_xlabel("x")
    ax.set_ylabel("z (Chebyshev)")

    # ax = fig.add_subplot(gs[1, 0])
    # mesh = ax.pcolormesh(kx, kz, np.int32(lowpass_mask), shading='nearest')
    # plt.colorbar(mesh)
    # ax.set_title("k-space lowpass mask")

    # ax = fig.add_subplot(gs[1, 1])
    # mesh = ax.pcolormesh(kx, kz, np.int32(highpass_mask), shading='nearest')
    # plt.colorbar(mesh)
    # ax.set_title("k-space highpass mask")

    ax = fig.add_subplot(gs[0, 3])
    mesh = ax.pcolormesh(x, z_cheb, data_filter_low.T, shading='nearest')
    plt.colorbar(mesh)
    ax.set_title("lowpass filtered")
    ax.set_xlabel("x")
    ax.set_ylabel("z (Chebyshev)")

    ax = fig.add_subplot(gs[0, 4])
    mesh = ax.pcolormesh(x, z_cheb, data_filter_high.T, shading='nearest')
    plt.colorbar(mesh)
    ax.set_title("highpass filtered")
    ax.set_xlabel("x")
    ax.set_ylabel("z (Chebyshev)")

    ##################################
    # 1st row, interpolated data, FFT w.r.t. z coordinates

    ax = fig.add_subplot(gs[1, 0])
    mesh = ax.pcolormesh(x, z_lin, data_interp.T, shading='nearest')
    plt.colorbar(mesh)
    ax.set_title("Interpolated data")
    ax.set_xlabel("x")
    ax.set_ylabel("z (Linear)")

    ax = fig.add_subplot(gs[1, 1])
    mesh = ax.pcolormesh(kx, kz, data_interp_fft_abs.T, shading='nearest')
    plt.colorbar(mesh)
    ax.set_title("FFT (magnitude)")
    ax.set_xlabel("kx")
    ax.set_ylabel("kz")

    ax = fig.add_subplot(gs[1, 2])
    mesh = ax.pcolormesh(x, z_lin, (data_interp_fft_inv - data_interp).T, shading='nearest')
    plt.colorbar(mesh)
    ax.set_title("Inverse of FFT minus untransformed")
    ax.set_xlabel("x")
    ax.set_ylabel("z (Linear)")

    ax = fig.add_subplot(gs[1, 3])
    mesh = ax.pcolormesh(x, z_lin, data_interp_filter_low.T, shading='nearest')
    plt.colorbar(mesh)
    ax.set_title("lowpass filtered")
    ax.set_xlabel("x")
    ax.set_ylabel("z (Linear)")

    ax = fig.add_subplot(gs[1, 4])
    mesh = ax.pcolormesh(x, z_lin, data_interp_filter_high.T, shading='nearest')
    plt.colorbar(mesh)
    ax.set_title("highpass filtered")
    ax.set_xlabel("x")
    ax.set_ylabel("z (Linear)")

    fig.tight_layout()
    plt.show()

def test_dedalus_interp():
    print("Running interpolation test case")
    Lx = 5
    Lz = 2
    gs = plt.GridSpec(3, 3)
    fig = plt.figure(figsize=(gs.ncols * 4, gs.nrows * 3))
    fig.suptitle("Interpolation of 3-D grids from Chebyshev to Fourier z axis\nPlotted averaged over y")

    res = 50
    res_z_interp = 100

    x = y = de.Fourier('x', res, interval=(0, Lx)).grid()
    z = de.Chebyshev('z', res, interval=(0, Lz)).grid()
    z_lin = de.Fourier('z', res_z_interp, interval=(0, Lz)).grid()

    arr = np.zeros((res, res, res))
    if PerlinNoise is not None:
        print("  Generating 3-D perlin noise for testing...")
        noise = PerlinNoise(octaves=2, seed=np.random.rand() * 9999999)
        for i in range(res):
            for j in range(res):
                for k in range(res):
                    arr[i, j, k] = noise([x[i], y[j], z[k]])
    else:
        xgrid = np.add.outer(np.add.outer(x, empty), empty)
        ygrid = np.add.outer(np.add.outer(empty, y), empty)
        zgrid = np.add.outer(np.add.outer(empty, empty), z)
        arr = np.sin(8 * np.pi * zgrid / L) + np.sin(4 * np.pi * xgrid / L)

    arr_dz = np.gradient(arr, z, axis=-1, edge_order=2)
    arr_dz2 = np.gradient(arr_dz, z, axis=-1, edge_order=2)

    print("  Interpolating...")
    start_time = time.time()
    arr_interp = interp_to_basis(arr, axis=-1, src=de.Chebyshev, dest=de.Fourier, dest_res=res_z_interp)
    duration = time.time() - start_time
    arr_interp_dz = np.gradient(arr_interp, z_lin, axis=-1, edge_order=2)
    arr_interp_dz2 = np.gradient(arr_interp_dz, z_lin, axis=-1, edge_order=2)

    print("  Interpolating back...")
    arr_interp2 = interp_to_basis(arr_interp, axis=-1, src=de.Fourier, dest=de.Chebyshev, dest_res=res)
    arr_interp2_dz = np.gradient(arr_interp2, z, axis=-1, edge_order=2)
    arr_interp2_dz2 = np.gradient(arr_interp2_dz, z, axis=-1, edge_order=2)

    ax = fig.add_subplot(gs[0, 0])
    mesh = ax.pcolormesh(x, z, np.mean(arr, axis=1).T, shading='nearest')
    plt.colorbar(mesh)
    ax.set_title("Original data ({}x{}x{})".format(res, res, res))
    ax.set_xlabel("x")
    ax.set_ylabel("z (Chebyshev)")

    ax = fig.add_subplot(gs[0, 1])
    mesh = ax.pcolormesh(x, z, np.mean(arr_dz, axis=1).T, shading='nearest')
    plt.colorbar(mesh)
    ax.set_title("dz(Original data)")
    ax.set_xlabel("x")
    ax.set_ylabel("z (Chebyshev)")

    ax = fig.add_subplot(gs[0, 2])
    mesh = ax.pcolormesh(x, z, np.mean(arr_dz2, axis=1).T, shading='nearest')
    plt.colorbar(mesh)
    ax.set_title("dz^2(Original data)")
    ax.set_xlabel("x")
    ax.set_ylabel("z (Chebyshev)")

    ax = fig.add_subplot(gs[1, 0])
    mesh = ax.pcolormesh(x, z_lin, np.mean(arr_interp, axis=1).T, shading='nearest')
    plt.colorbar(mesh)
    ax.set_title("Interpolated to ({}x{}x{}) in {}s".format(res, res, res_z_interp, np.round(duration, 3)))
    ax.set_xlabel("x")
    ax.set_ylabel("z (Fourier)")

    ax = fig.add_subplot(gs[1, 1])
    mesh = ax.pcolormesh(x, z_lin, np.mean(arr_interp_dz, axis=1).T, shading='nearest')
    plt.colorbar(mesh)
    ax.set_title("dz(Interpolated)")
    ax.set_xlabel("x")
    ax.set_ylabel("z (Fourier)")

    ax = fig.add_subplot(gs[1, 2])
    mesh = ax.pcolormesh(x, z_lin, np.mean(arr_interp_dz2, axis=1).T, shading='nearest')
    plt.colorbar(mesh)
    ax.set_title("dz^2(Interpolated)")
    ax.set_xlabel("x")
    ax.set_ylabel("z (Fourier)")

    ax = fig.add_subplot(gs[2, 0])
    mesh = ax.pcolormesh(x, z, np.mean(arr_interp2 - arr, axis=1).T, shading='nearest')
    plt.colorbar(mesh)
    ax.set_title("Un-interpolated - Original")
    ax.set_xlabel("x")
    ax.set_ylabel("z (Chebyshev)")

    ax = fig.add_subplot(gs[2, 1])
    mesh = ax.pcolormesh(x, z, np.mean(arr_interp2_dz - arr_dz, axis=1).T, shading='nearest')
    plt.colorbar(mesh)
    ax.set_title("dz(Un-interpolated) - dz(Original)")
    ax.set_xlabel("x")
    ax.set_ylabel("z (Chebyshev)")

    ax = fig.add_subplot(gs[2, 2])
    mesh = ax.pcolormesh(x, z, np.mean(arr_interp2_dz2 - arr_dz2, axis=1).T, shading='nearest')
    plt.colorbar(mesh)
    ax.set_title("dz^2(Un-interpolated) - dz^2(Original)")
    ax.set_xlabel("x")
    ax.set_ylabel("z (Chebyshev)")

    plt.tight_layout()
    plt.show()

# If you run this script as a program from the command line, run all the tests
if __name__ == '__main__':
    plt.rcParams.update({'font.size': 8})
    plot_test_case()
    test_dedalus_interp()
    time_execution() 
