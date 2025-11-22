import numpy as np
import matplotlib.pyplot as plt


def compare_rcs_cpu_gpu(theta_cpu_deg,
                        rcs_cpu_dB,
                        theta_gpu_deg,
                        rcs_gpu_dB,
                        opts=None):
    """
    Compare CPU vs GPU RCS angular cuts.

    Parameters
    ----------
    theta_cpu_deg : array_like
        1D array of angles (deg) for CPU solution.
    rcs_cpu_dB : array_like
        1D array of RCS values (dB) from CPU.
    theta_gpu_deg : array_like
        1D array of angles (deg) for GPU solution.
    rcs_gpu_dB : array_like
        1D array of RCS values (dB) from GPU.
    opts : dict, optional
        Options dictionary with keys:
            'interp_method' : str, interpolation kind for np.interp or np.interp-like.
                              Supported: 'linear' (default).
            'title'         : str, title prefix for plots.
            'unwrap'        : bool, if True, unwrap angles in degrees.

    Returns
    -------
    stats : dict
        {
            'theta_deg'      : common angle grid,
            'rcs_cpu_dB'     : CPU RCS on common grid,
            'rcs_gpu_dB'     : GPU RCS interpolated onto common grid,
            'diff_dB'        : GPU - CPU difference (dB),
            'maxAbsDiff_dB'  : max |diff_dB|,
            'meanAbsDiff_dB' : mean |diff_dB|,
            'rmsDiff_lin'    : RMS difference in linear RCS units
        }
    """
    if opts is None:
        opts = {}

    interp_method = opts.get('interp_method', 'linear')
    title = opts.get('title', 'CPU vs GPU RCS')
    unwrap = opts.get('unwrap', False)

    # --- Normalize inputs ---
    theta_cpu_deg = _normalize_angle_input(theta_cpu_deg, 'theta_cpu_deg')
    theta_gpu_deg = _normalize_angle_input(theta_gpu_deg, 'theta_gpu_deg')
    rcs_cpu_dB = _normalize_numeric_input(rcs_cpu_dB, 'rcs_cpu_dB')
    rcs_gpu_dB = _normalize_numeric_input(rcs_gpu_dB, 'rcs_gpu_dB')

    # --- Optional unwrap in degrees ---
    if unwrap:
        theta_cpu_deg = _unwrap_deg(theta_cpu_deg)
        theta_gpu_deg = _unwrap_deg(theta_gpu_deg)

    # --- Interpolation to common grid ---
    # We'll use CPU angles as the reference grid.
    if (theta_cpu_deg.shape != theta_gpu_deg.shape) or not np.allclose(theta_cpu_deg, theta_gpu_deg):
        # Sort GPU angles and interpolate GPU RCS onto CPU angle grid
        sort_idx = np.argsort(theta_gpu_deg)
        theta_gpu_sorted = theta_gpu_deg[sort_idx]
        rcs_gpu_sorted = rcs_gpu_dB[sort_idx]

        if interp_method.lower() == 'linear':
            rcs_gpu_interp = np.interp(theta_cpu_deg, theta_gpu_sorted, rcs_gpu_sorted)
        else:
            raise ValueError(f"Unsupported interp_method: {interp_method}")

        theta_common_deg = theta_cpu_deg
        rcs_cpu_common_dB = rcs_cpu_dB
        rcs_gpu_common_dB = rcs_gpu_interp
    else:
        theta_common_deg = theta_cpu_deg
        rcs_cpu_common_dB = rcs_cpu_dB
        rcs_gpu_common_dB = rcs_gpu_dB

    # --- Difference calculation in dB ---
    diff_dB = rcs_gpu_common_dB - rcs_cpu_common_dB

    # Convert to linear RCS for RMS difference
    sigma_cpu_lin = 10.0 ** (rcs_cpu_common_dB / 10.0)
    sigma_gpu_lin = 10.0 ** (rcs_gpu_common_dB / 10.0)
    diff_lin = sigma_gpu_lin - sigma_cpu_lin

    maxAbsDiff_dB = float(np.max(np.abs(diff_dB)))
    meanAbsDiff_dB = float(np.mean(np.abs(diff_dB)))
    rmsDiff_lin = float(np.sqrt(np.mean(diff_lin ** 2)))

    stats = {
        'theta_deg': theta_common_deg,
        'rcs_cpu_dB': rcs_cpu_common_dB,
        'rcs_gpu_dB': rcs_gpu_common_dB,
        'diff_dB': diff_dB,
        'maxAbsDiff_dB': maxAbsDiff_dB,
        'meanAbsDiff_dB': meanAbsDiff_dB,
        'rmsDiff_lin': rmsDiff_lin,
    }

    # --- Plots ---

    # 1) Overlay RCS patterns
    plt.figure()
    plt.plot(theta_common_deg, rcs_cpu_common_dB, label='CPU', linewidth=1.5)
    plt.plot(theta_common_deg, rcs_gpu_common_dB, '--', label='GPU', linewidth=1.5)
    plt.grid(True)
    plt.xlabel(r'$\theta$ (deg)')
    plt.ylabel('RCS (dB)')
    plt.title(f'{title} - RCS Comparison')
    plt.legend()
    plt.tight_layout()

    # 2) Difference vs angle
    plt.figure()
    plt.plot(theta_common_deg, diff_dB, linewidth=1.5)
    plt.grid(True)
    plt.xlabel(r'$\theta$ (deg)')
    plt.ylabel(r'$\Delta$RCS = GPU - CPU (dB)')
    plt.title(f'{title} - Difference (dB)')
    plt.tight_layout()

    # 3) Histogram of differences
    plt.figure()
    plt.hist(diff_dB, bins=40)
    plt.grid(True)
    plt.xlabel(r'$\Delta$RCS (dB)')
    plt.ylabel('Count')
    plt.title(f'{title} - Histogram of Differences')
    plt.tight_layout()

    print('--- CPU vs GPU RCS Comparison ---')
    print(f'Max |GPU - CPU|     : {maxAbsDiff_dB:.3f} dB')
    print(f'Mean |GPU - CPU|    : {meanAbsDiff_dB:.3f} dB')
    print(f'RMS difference (lin): {rmsDiff_lin:.3e} (RCS units)')

    return stats


def _normalize_angle_input(x, name='angle'):
    """
    Normalize angle-like input to a clean 1D float numpy array.
    Remove NaNs, take real part if complex.
    """
    x = np.asarray(x)
    if x.ndim == 0:
        x = x.reshape(1)
    x = x.astype(float)  # will raise if non-numeric

    # Flatten to 1D
    x = x.ravel()

    # Remove NaNs
    mask = ~np.isnan(x)
    if not np.any(mask):
        raise ValueError(f'Input {name} contains only NaNs.')
    x = x[mask]

    return x


def _normalize_numeric_input(x, name='value'):
    """
    Normalize numeric input to a clean 1D float numpy array.
    """
    x = np.asarray(x)
    if x.ndim == 0:
        x = x.reshape(1)
    x = x.astype(float)
    return x.ravel()


def _unwrap_deg(ang_deg):
    """
    Unwrap angles in degrees (like MATLAB unwrap but in deg).
    """
    ang_rad = np.deg2rad(ang_deg)
    ang_unwrapped_rad = np.unwrap(ang_rad)
    return np.rad2deg(ang_unwrapped_rad)
