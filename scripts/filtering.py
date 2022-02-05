#!/usr/bin/env python3

import numpy as np
from dedalus import public as de
from matplotlib import pyplot as plt
import scipy.fft as fft
import scipy.interpolate as interpolate

def interp_to_bases(arr, bases, newbases):
    """
    Interpolates the n-dimensional array `arr` along `axis` using `newbasis`
    """

    # The need for this function might be hard to explain. Here's my attempt.

    # In order to FFT a field from dedalus, we need to provide scipy with the sampling
    # interval (i.e. grid spacing) of the data. This allows scipy to calculate the
    # bases in k-space (e.g. kx, the wavenumber) that we would use to plot the transformed
    # data.

    # This is where we encounter a problem: the basis for our z-axis is a Chebyshev
    # polynomial. This basis has more points towards each end of the axis. This means that
    # the sampling interval (i.e. grid spacing) of the data in the z-direction is not
    # constant.
    # See: https://dedalus-project.readthedocs.io/en/latest/notebooks/dedalus_tutorial_bases_domains.html#Basis-grids-and-scale-factors
    # FFTs have no way of dealing with this (as far as I can find).

    # Hence, I decided to create this function to interpolate the data such that it
    # can be plotted against a z-axis with a constant grid spacing. This way, we can
    # interpolate the data, FFT, then interpolate the data back to the original Chebyshev
    # basis if necessary.

    # Essentially, after running the array through this function, it can be plotted against
    # `newbases` instead of `bases`

    # -- end explanation --

    # Change the bases into an array of coordinates specifying every point in arr
    # Just the format that LinearNDInterpolator wants
    points = np.stack(np.meshgrid(*bases), axis=-1)
    points = points.reshape(np.product(arr.shape), points.shape[-1])

    # Interpolate using the desired basis
    # See the example at:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LinearNDInterpolator.html
    arr_interp = interpolate.LinearNDInterpolator(points, arr.flatten(), fill_value=0)(*np.meshgrid(*newbases))
    return arr_interp


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
    return np.array(newbases)
    

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


def kspace_lowpass(data, axes, bases, lambda_cutoff, interp=False):
    """
    Apply a lowpass filter to `data` along the specified axes.
    `bases` is the set of bases that data is specified by (e.g. x, y, and z arrays).
    `lambda_cutoff` is the smallest wavelength (equivalent to the largest frequency) that will be retained.
    """

    # Interpolate to linear axis scales
    bases_lin = create_linear_bases(bases)
    data_interp = interp_to_bases(data, bases, bases_lin) if interp else data

    # Transform to k space
    data_fft, kbases = fft_nd(data_interp, axes, bases_lin)

    # k_space_magnitude[i, j] = kz[i]^2 + kx[j]^2
    # i.e. each element is the magnitude of the corresponding wave-vector
    k_space_magnitude = np.add.outer(*[basis * basis for basis in kbases])

    # the masks are 2d arrays of booleans that can be used to select only low/high frequencies from a transformed array
    cutoff = 2 * np.pi / lambda_cutoff
    lowpass_mask = k_space_magnitude < cutoff*cutoff

    # Apply the mask
    data_fft = np.where(lowpass_mask, data_fft, 0)

    # Invert the FFT back to real space
    data_fft_reversed = fft_nd_inverse(data_fft, axes)

    # Interpolate back to the original basis
    return interp_to_bases(data_fft_reversed, bases_lin, bases) if interp else data_fft_reversed


def kspace_highpass(data, axes, bases, lambda_cutoff, interp=False):
    """
    Apply a highpass filter to `data` along the specified axes.
    `bases` is the set of bases that data is specified by (e.g. x, y, and z arrays).
    `lambda_cutoff` is the largest wavelength (equivalent to the smallest frequency) that will be retained.
    """

    # Interpolate to linear axis scales
    bases_lin = create_linear_bases(bases)
    data_interp = interp_to_bases(data, bases, bases_lin) if interp else data

    # Transform to k space
    data_fft, kbases = fft_nd(data_interp, axes, bases_lin)

    # k_space_magnitude[i, j] = kz[i]^2 + kx[j]^2
    # i.e. each element is the magnitude of the corresponding wave-vector
    k_space_magnitude = np.add.outer(*[basis * basis for basis in kbases])

    # the masks are 2d arrays of booleans that can be used to select only low/high frequencies from a transformed array
    cutoff = 2 * np.pi / lambda_cutoff
    highpass_mask = k_space_magnitude >= cutoff*cutoff

    # Apply the mask
    data_fft = np.where(highpass_mask, data_fft, 0)

    # Invert the FFT back to real space
    data_fft_reversed = fft_nd_inverse(data_fft, axes)

    # Interpolate back to the original basis
    return interp_to_bases(data_fft_reversed, bases_lin, bases) if interp else data_fft_reversed


def plot_test_case():
    """
    Plots a test case of the filtering functions in this file
    """

    print("Running filtering test case")

    print("  Creating data...")
    Lx = 1
    Lz = 1
    resX = 50
    resZ = 50
    wavelength_cutoff = Lx / 2 # Radius in wavelength-space

    x = np.arange(0, Lx, Lx/resX)
    z_cheb = de.Chebyshev('z', resZ, interval=(0, Lz), dealias=2).grid(scale=1)

    data = np.sin(8 * np.pi * np.add.outer(z_cheb, np.zeros(len(x))) / Lz) + np.sin(2 * np.pi * np.add.outer(np.zeros(len(z_cheb)), x) / Lx)
    (x, z_lin) = create_linear_bases((x, z_cheb))
    data_interp = interp_to_bases(data, (x, z_cheb), (x, z_lin))

    print("  Calculating transforms...")
    data_fft, (kx, kz) = fft_nd(data, (0, 1), (x, z_lin))
    data_fft_abs = np.abs(data_fft)
    data_fft_inv = fft_nd_inverse(data_fft, (0, 1))
    data_filter_low = kspace_lowpass(data, (0, 1), (x, z_lin), wavelength_cutoff, interp=True)
    data_filter_high = kspace_highpass(data, (0, 1), (x, z_lin), wavelength_cutoff, interp=True)

    data_interp_fft, (kx, kz) = fft_nd(data_interp, (0, 1), (x, z_lin))
    data_interp_fft_abs = np.abs(data_interp_fft)
    data_interp_fft_inv = fft_nd_inverse(data_interp_fft, (0, 1))
    data_interp_filter_low = kspace_lowpass(data, (0, 1), (x, z_cheb), wavelength_cutoff, interp=True)
    data_interp_filter_high = kspace_highpass(data, (0, 1), (x, z_cheb), wavelength_cutoff, interp=True)

    print("  Plotting...")
    gs = plt.GridSpec(2, 5)
    fig = plt.figure(figsize=(gs.ncols * 4, gs.nrows * 3))
    fig.suptitle(
        "High- and low-pass filtering with and without interpolation.\n"
        "The first row shows filtering the simple (but incorrect) way, simply doing the FFT using a linear basis instead of the Chebyshev one.\n"
        "The second row is the same process, but the z-axis is interpolated into a linear basis first.\n"
        "Note the noisy (and incorrect) frequencies for the z-axis in the first case, and how they're mostly fixed in the second."
    )

    ##################################
    # 1st row, original data, FFT of original data w.r.t. linear basis (clearly wrong approach)

    ax = fig.add_subplot(gs[0, 0])
    mesh = ax.pcolormesh(x, z_cheb, data, shading='nearest')
    plt.colorbar(mesh)
    ax.set_title("Original data: sin(8πz/Lz) + sin(2πx/Lx)")
    ax.set_xlabel("x")
    ax.set_ylabel("z (Chebyshev)")

    ax = fig.add_subplot(gs[0, 1])
    mesh = ax.pcolormesh(kx, kz, data_fft_abs, shading='nearest')
    plt.colorbar(mesh)
    ax.set_title("FFT (magnitude) using linear basis")
    ax.set_xlabel("kx")
    ax.set_ylabel("kz")

    ax = fig.add_subplot(gs[0, 2])
    mesh = ax.pcolormesh(x, z_cheb, data_fft_inv - data, shading='nearest')
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
    mesh = ax.pcolormesh(x, z_cheb, data_filter_low, shading='nearest')
    plt.colorbar(mesh)
    ax.set_title("lowpass filtered")
    ax.set_xlabel("x")
    ax.set_ylabel("z (Chebyshev)")

    ax = fig.add_subplot(gs[0, 4])
    mesh = ax.pcolormesh(x, z_cheb, data_filter_high, shading='nearest')
    plt.colorbar(mesh)
    ax.set_title("highpass filtered")
    ax.set_xlabel("x")
    ax.set_ylabel("z (Chebyshev)")

    ##################################
    # 1st row, interpolated data, FFT w.r.t. z coordinates

    ax = fig.add_subplot(gs[1, 0])
    mesh = ax.pcolormesh(x, z_lin, data_interp, shading='nearest')
    plt.colorbar(mesh)
    ax.set_title("Interpolated data")
    ax.set_xlabel("x")
    ax.set_ylabel("z (Linear)")

    ax = fig.add_subplot(gs[1, 1])
    mesh = ax.pcolormesh(kx, kz, data_interp_fft_abs, shading='nearest')
    plt.colorbar(mesh)
    ax.set_title("FFT (magnitude)")
    ax.set_xlabel("kx")
    ax.set_ylabel("kz")

    ax = fig.add_subplot(gs[1, 2])
    mesh = ax.pcolormesh(x, z_lin, data_interp_fft_inv - data_interp, shading='nearest')
    plt.colorbar(mesh)
    ax.set_title("Inverse of FFT minus untransformed")
    ax.set_xlabel("x")
    ax.set_ylabel("z (Linear)")

    ax = fig.add_subplot(gs[1, 3])
    mesh = ax.pcolormesh(x, z_cheb, data_interp_filter_low, shading='nearest')
    plt.colorbar(mesh)
    ax.set_title("lowpass filtered")
    ax.set_xlabel("x")
    ax.set_ylabel("z (Chebyshev)")

    ax = fig.add_subplot(gs[1, 4])
    mesh = ax.pcolormesh(x, z_cheb, data_interp_filter_high, shading='nearest')
    plt.colorbar(mesh)
    ax.set_title("highpass filtered")
    ax.set_xlabel("x")
    ax.set_ylabel("z (Chebyshev)")

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    plt.rcParams.update({'font.size': 8})
    plot_test_case()

