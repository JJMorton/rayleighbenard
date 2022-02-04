import json
import numpy as np
import scipy.fft as fft

def calc_plot_size(params):
    figwidth = min(14, 4 * params['Lx'] / params['Lz'])
    figheight = figwidth * params['Lz'] / params['Lx']
    return figwidth, figheight

def read_params(data_dir="analysis"):
    with open(f'{data_dir}/params.json', 'r') as f:
        params = json.load(f)
    return params

def save_params(params, data_dir="analysis"):
    with open(f'{data_dir}/params.json', 'w') as f:
        json.dump(params, f, indent=2)

def create_params_string(params):
    return "Ra = {:.1E}, Pr = {}, Ek = {}, Theta = {:.3f}".format(params["Ra"], params["Pr"], params["Ek"], params["Theta"])

def scale_has_uniform_spacing(scale):
        epsilon = 1e-15
        # Calculates array of differences between adjacent elements, subtracts the difference between the first
        # two elements, and tests if these are all basically zero
        return np.all(np.abs((scale[1:] - scale[:-1]) - (scale[1] - scale[0])) < epsilon)

def fft_nd(data, axes, scales):
    """
    Takes the N-dimensional fourier transform of the array `data`
    `axes` is a list of indices specifying the dimensions of `data` to transform
    `scales` is a list of the bases (i.e. what you would plot `data` against, like x, y or z)
    """
    if len(axes) != len(scales):
        raise Exception("fft_nd: Number of axes must equal number of scales")

    # Make sure the axes are uniformly spaced all the way along
    for scale in scales:
        if not scale_has_uniform_spacing(scale):
            raise Exception("Attempt to apply an FFT to an axis with non-uniform spacing")

    # Sampling interval of each axis
    spacing = [ scale[1] - scale[0] for scale in scales ]

    # Calculate the wavenumbers to plot against in k-space
    # fftfreq returns frequencies, we want wavenumbers (i.e. 2*pi*f)
    kscales = [ 2 * np.pi * fft.fftshift(fft.fftfreq(len(scale), d=d)) for (scale, d) in zip(scales, spacing) ]

    # Actually perform the FFT
    transformed = fft.fftshift(fft.fftn(data, axes=axes))

    return transformed, kscales


def fft_nd_inverse(data, axes):
    return np.real(fft.ifftn(fft.ifftshift(data), axes=axes))


def kspace_lowpass(data, axes, scales, lambda_cutoff):
    # Transform to k space
    data_fft, (kx1, kx2) = fft_nd(data, axes, scales)

    # k_space_magnitude[i, j] = kz[i]^2 + kx[j]^2
    # i.e. each element is the magnitude of the corresponding wave-vector
    k_space_magnitude = np.add.outer(kx2*kx2, kx1*kx1)

    # the masks are 2d arrays of booleans that can be used to select only low/high frequencies from a transformed array
    cutoff = 2 * np.pi / lambda_cutoff
    lowpass_mask = k_space_magnitude < cutoff*cutoff

    # Apply the mask
    data_fft = np.where(lowpass_mask, data_fft, 0)

    # Invert the FFT
    return fft_nd_inverse(data_fft, axes)


def kspace_highpass(data, axes, scales, lambda_cutoff):
    # Transform to k space
    data_fft, (kx1, kx2) = fft_nd(data, axes, scales)

    # k_space_magnitude[i, j] = kz[i]^2 + kx[j]^2
    # i.e. each element is the magnitude of the corresponding wave-vector
    k_space_magnitude = np.add.outer(kx2*kx2, kx1*kx1)

    # the masks are 2d arrays of booleans that can be used to select only low/high frequencies from a transformed array
    cutoff = 2 * np.pi / lambda_cutoff
    highpass_mask = k_space_magnitude >= cutoff*cutoff

    # Apply the mask
    data_fft = np.where(highpass_mask, data_fft, 0)

    # Invert the FFT
    return fft_nd_inverse(data_fft, axes)


