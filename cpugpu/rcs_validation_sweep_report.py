"""
Multi-frequency RCS CPU vs GPU Validation Script
------------------------------------------------
- Loads CPU & GPU RCS results from CSV (multi-frequency)
- Computes accuracy metrics per frequency up to a max freq (e.g., 50 GHz)
- Generates a multi-page PDF report with:
    * Summary / title page
    * Metrics vs frequency plots
    * Example RCS overlay & error plots at a selected frequency

Requirements:
    pip install numpy matplotlib pandas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import datetime

# ------------------------------------------------------------
# CONFIGURATION (EDIT THIS BLOCK FOR YOUR USE CASE)
# ------------------------------------------------------------

CPU_CSV_PATH = "rcs_cpu_multi.csv"
GPU_CSV_PATH = "rcs_gpu_multi.csv"
OUTPUT_PDF   = "rcs_validation_sweep_report.pdf"

META = {
    "project_name": "RCS Cone Validation (Frequency Sweep)",
    "target_description": "1 m cone, 0.5 m base, 0.1 m tip radius",
    "solver_cpu": "MoM/PO - CPU Baseline",
    "solver_gpu": "MoM/PO - GPU Accelerated",
    "cpu_hardware": "Xeon Gold (example)",
    "gpu_hardware": "NVIDIA A100 (example)",
    "polarization": "HH",
}

# Only evaluate frequencies up to this value (GHz)
FREQ_MAX_GHZ = 50.0
FREQ_MIN_GHZ = 0.0  # Set >0 if you want to ignore very low freqs

# Set to True if your theta wraps around and you want to unwrap it
UNWRAP_ANGLES = False


# ------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------

def load_rcs_multifreq_csv(path):
    """
    Load multi-frequency RCS data from a CSV file.

    Expected columns:
        freq_ghz, theta_deg, rcs_db
    (You can adjust the column name lists below.)

    Returns
    -------
    df : pandas.DataFrame
        columns: freq_ghz, theta_deg, rcs_db
    """
    df = pd.read_csv(path)

    # Allow some flexibility in column names
    freq_cols  = ["freq_ghz", "frequency_ghz", "freq", "f_ghz"]
    theta_cols = ["theta_deg", "theta", "angle_deg", "angle"]
    rcs_cols   = ["rcs_db", "rcs_dB", "rcs", "RCS_dB"]

    freq_col  = _find_column(df, freq_cols, "frequency")
    theta_col = _find_column(df, theta_cols, "theta")
    rcs_col   = _find_column(df, rcs_cols, "rcs")

    df = df[[freq_col, theta_col, rcs_col]].copy()
    df.columns = ["freq_ghz", "theta_deg", "rcs_db"]  # normalize

    df["freq_ghz"]   = df["freq_ghz"].astype(float)
    df["theta_deg"]  = df["theta_deg"].astype(float)
    df["rcs_db"]     = df["rcs_db"].astype(float)

    return df


def _find_column(df, candidates, logical_name):
    for c in df.columns:
        if c in candidates:
            return c
    raise ValueError(
        f"No {logical_name} column found. "
        f"Expected one of {candidates}, got {list(df.columns)}"
    )


# ------------------------------------------------------------
# CORE MATH / METRICS
# ------------------------------------------------------------

def unwrap_deg(ang_deg):
    """Unwrap angles in degrees (like MATLAB unwrap, but in deg)."""
    ang_rad = np.deg2rad(ang_deg)
    ang_unwrapped = np.unwrap(ang_rad)
    return np.rad2deg(ang_unwrapped)


def normalize_angle_input(x, name="angle"):
    """Convert to 1D float array, remove NaNs."""
    x = np.asarray(x, dtype=float).ravel()
    mask = ~np.isnan(x)
    if not np.any(mask):
        raise ValueError(f"Input {name} contains only NaNs.")
    return x[mask]


def normalize_numeric_input(x, name="value"):
    """Convert to 1D float array."""
    x = np.asarray(x, dtype=float).ravel()
    return x


def compute_rcs_metrics_single(theta_cpu_deg,
                               rcs_cpu_dB,
                               theta_gpu_deg,
                               rcs_gpu_dB,
                               unwrap=False):
    """
    Compute comparison metrics between CPU and GPU RCS curves at a single frequency.

    Returns
    -------
    result : dict with:
        theta_deg
        rcs_cpu_dB
        rcs_gpu_dB
        diff_dB
        maxAbsDiff_dB
        meanAbsDiff_dB
        p95AbsDiff_dB
        rmsDiff_lin
        nrmse_lin
        corr_coeff
    """
    # Normalize inputs
    theta_cpu_deg = normalize_angle_input(theta_cpu_deg, "theta_cpu_deg")
    theta_gpu_deg = normalize_angle_input(theta_gpu_deg, "theta_gpu_deg")
    rcs_cpu_dB    = normalize_numeric_input(rcs_cpu_dB, "rcs_cpu_dB")
    rcs_gpu_dB    = normalize_numeric_input(rcs_gpu_dB, "rcs_gpu_dB")

    if unwrap:
        theta_cpu_deg = unwrap_deg(theta_cpu_deg)
        theta_gpu_deg = unwrap_deg(theta_gpu_deg)

    # Interpolate GPU RCS onto CPU angle grid if grids differ
    if (theta_cpu_deg.shape != theta_gpu_deg.shape) or \
       (not np.allclose(theta_cpu_deg, theta_gpu_deg)):
        sort_idx = np.argsort(theta_gpu_deg)
        theta_gpu_sorted = theta_gpu_deg[sort_idx]
        rcs_gpu_sorted   = rcs_gpu_dB[sort_idx]

        # linear interpolation
        rcs_gpu_interp = np.interp(theta_cpu_deg, theta_gpu_sorted, rcs_gpu_sorted)
        theta_common_deg   = theta_cpu_deg
        rcs_cpu_common_dB  = rcs_cpu_dB
        rcs_gpu_common_dB  = rcs_gpu_interp
    else:
        theta_common_deg   = theta_cpu_deg
        rcs_cpu_common_dB  = rcs_cpu_dB
        rcs_gpu_common_dB  = rcs_gpu_dB

    # Difference in dB
    diff_dB = rcs_gpu_common_dB - rcs_cpu_common_dB

    # Linear-domain metrics
    sigma_cpu_lin = 10.0 ** (rcs_cpu_common_dB / 10.0)
    sigma_gpu_lin = 10.0 ** (rcs_gpu_common_dB / 10.0)
    diff_lin      = sigma_gpu_lin - sigma_cpu_lin

    rmsDiff_lin = float(np.sqrt(np.mean(diff_lin ** 2)))
    max_cpu_lin = float(np.max(sigma_cpu_lin))
    nrmse_lin   = float(rmsDiff_lin / max_cpu_lin) if max_cpu_lin > 0 else np.nan

    abs_diff_dB = np.abs(diff_dB)
    maxAbsDiff_dB  = float(np.max(abs_diff_dB))
    meanAbsDiff_dB = float(np.mean(abs_diff_dB))
    p95AbsDiff_dB  = float(np.percentile(abs_diff_dB, 95))

    # Pattern correlation in dB-domain
    x = rcs_cpu_common_dB - np.mean(rcs_cpu_common_dB)
    y = rcs_gpu_common_dB - np.mean(rcs_gpu_common_dB)
    denom = np.sqrt(np.sum(x**2) * np.sum(y**2))
    if denom > 0:
        corr_coeff = float(np.sum(x * y) / denom)
    else:
        corr_coeff = np.nan

    result = {
        "theta_deg":       theta_common_deg,
        "rcs_cpu_dB":      rcs_cpu_common_dB,
        "rcs_gpu_dB":      rcs_gpu_common_dB,
        "diff_dB":         diff_dB,
        "maxAbsDiff_dB":   maxAbsDiff_dB,
        "meanAbsDiff_dB":  meanAbsDiff_dB,
        "p95AbsDiff_dB":   p95AbsDiff_dB,
        "rmsDiff_lin":     rmsDiff_lin,
        "nrmse_lin":       nrmse_lin,
        "corr_coeff":      corr_coeff,
    }
    return result


# ------------------------------------------------------------
# PLOTTING / PDF REPORT
# ------------------------------------------------------------

def add_sweep_title_page(pdf, meta, global_metrics, freqs_ghz):
    """Create a title/summary page for the sweep."""
    fig = plt.figure(figsize=(8.5, 11))
    fig.clf()

    text = []

    text.append("RCS Solver Validation Report (Frequency Sweep)")
    text.append("")
    text.append(f"Date: {datetime.date.today().isoformat()}")
    text.append("")
    text.append(f"Project: {meta.get('project_name', 'N/A')}")
    text.append(f"Target:  {meta.get('target_description', 'N/A')}")
    text.append("")
    text.append(f"CPU Solver:   {meta.get('solver_cpu', 'N/A')}")
    text.append(f"CPU Hardware: {meta.get('cpu_hardware', 'N/A')}")
    text.append("")
    text.append(f"GPU Solver:   {meta.get('solver_gpu', 'N/A')}")
    text.append(f"GPU Hardware: {meta.get('gpu_hardware', 'N/A')}")
    text.append("")
    if len(freqs_ghz) > 0:
        f_min = float(np.min(freqs_ghz))
        f_max = float(np.max(freqs_ghz))
        text.append(f"Frequency range analyzed: {f_min:.3f} – {f_max:.3f} GHz")
    else:
        text.append("Frequency range analyzed: (none)")
    text.append(f"Polarization: {meta.get('polarization', 'N/A')}")
    text.append("")
    text.append("Global Summary of Key Metrics (across frequencies):")
    text.append(f"  Max of Max |ΔRCS|     : {global_metrics['global_max_maxAbsDiff_dB']:.3f} dB")
    text.append(f"  Mean of Max |ΔRCS|    : {global_metrics['global_mean_maxAbsDiff_dB']:.3f} dB")
    text.append(f"  Max of Mean |ΔRCS|    : {global_metrics['global_max_meanAbsDiff_dB']:.3f} dB")
    text.append(f"  Mean Pattern Corr.    : {global_metrics['global_mean_corr_coeff']:.5f}")
    text.append(f"  Min Pattern Corr.     : {global_metrics['global_min_corr_coeff']:.5f}")

    full_text = "\n".join(text)
    plt.text(0.05, 0.95, full_text,
             va="top", ha="left", fontsize=11, family="monospace")
    plt.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def add_metrics_vs_freq_page(pdf, freqs_ghz, metrics_by_freq):
    """Plot key metrics vs frequency."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    ax1, ax2, ax3, ax4 = axes.flatten()

    maxAbsDiff_dB = np.array([m["maxAbsDiff_dB"] for m in metrics_by_freq])
    meanAbsDiff_dB = np.array([m["meanAbsDiff_dB"] for m in metrics_by_freq])
    corr_coeff = np.array([m["corr_coeff"] for m in metrics_by_freq])
    nrmse_lin = np.array([m["nrmse_lin"] for m in metrics_by_freq])

    ax1.plot(freqs_ghz, maxAbsDiff_dB, marker='o')
    ax1.set_title("Max |ΔRCS| vs Frequency")
    ax1.set_xlabel("Frequency (GHz)")
    ax1.set_ylabel("Max |ΔRCS| (dB)")
    ax1.grid(True)

    ax2.plot(freqs_ghz, meanAbsDiff_dB, marker='o')
    ax2.set_title("Mean |ΔRCS| vs Frequency")
    ax2.set_xlabel("Frequency (GHz)")
    ax2.set_ylabel("Mean |ΔRCS| (dB)")
    ax2.grid(True)

    ax3.plot(freqs_ghz, corr_coeff, marker='o')
    ax3.set_title("Pattern Correlation vs Frequency")
    ax3.set_xlabel("Frequency (GHz)")
    ax3.set_ylabel("Correlation Coefficient")
    ax3.grid(True)

    ax4.plot(freqs_ghz, nrmse_lin, marker='o')
    ax4.set_title("NRMSE (Linear) vs Frequency")
    ax4.set_xlabel("Frequency (GHz)")
    ax4.set_ylabel("NRMSE")
    ax4.grid(True)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def add_overlay_plot(pdf, theta_deg, rcs_cpu_dB, rcs_gpu_dB, freq_ghz):
    """CPU vs GPU RCS overlay plot for a specific frequency."""
    fig = plt.figure(figsize=(8.5, 5))
    plt.plot(theta_deg, rcs_cpu_dB, label="CPU", linewidth=1.5)
    plt.plot(theta_deg, rcs_gpu_dB, "--", label="GPU", linewidth=1.5)
    plt.grid(True)
    plt.xlabel("Angle (deg)")
    plt.ylabel("RCS (dB)")
    plt.title(f"CPU vs GPU RCS Comparison @ {freq_ghz:.3f} GHz")
    plt.legend()
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def add_diff_plot(pdf, theta_deg, diff_dB, freq_ghz):
    """ΔRCS vs angle plot for a specific frequency."""
    fig = plt.figure(figsize=(8.5, 5))
    plt.plot(theta_deg, diff_dB, linewidth=1.5)
    plt.grid(True)
    plt.xlabel("Angle (deg)")
    plt.ylabel("ΔRCS = GPU - CPU (dB)")
    plt.title(f"Difference Plot (ΔRCS vs Angle) @ {freq_ghz:.3f} GHz")
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def add_hist_plot(pdf, diff_dB, freq_ghz):
    """Histogram of ΔRCS at a specific frequency."""
    fig = plt.figure(figsize=(8.5, 5))
    plt.hist(diff_dB, bins=40)
    plt.grid(True)
    plt.xlabel("ΔRCS (dB)")
    plt.ylabel("Count")
    plt.title(f"Histogram of ΔRCS (GPU - CPU) @ {freq_ghz:.3f} GHz")
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ------------------------------------------------------------
# MAIN ENTRY POINT
# ------------------------------------------------------------

def main():
    # Load multi-frequency data
    df_cpu = load_rcs_multifreq_csv(CPU_CSV_PATH)
    df_gpu = load_rcs_multifreq_csv(GPU_CSV_PATH)

    # Get unique frequencies present in both CPU & GPU datasets
    freqs_cpu = set(df_cpu["freq_ghz"].unique())
    freqs_gpu = set(df_gpu["freq_ghz"].unique())
    freqs_common = sorted(list(freqs_cpu.intersection(freqs_gpu)))

    # Filter by requested frequency range
    freqs_common = [f for f in freqs_common if (FREQ_MIN_GHZ <= f <= FREQ_MAX_GHZ)]

    if not freqs_common:
        raise RuntimeError("No common frequencies in the specified range.")

    metrics_by_freq = []
    freqs_used = []

    # Compute metrics per frequency
    for f in freqs_common:
        sub_cpu = df_cpu[df_cpu["freq_ghz"] == f]
        sub_gpu = df_gpu[df_gpu["freq_ghz"] == f]

        theta_cpu = sub_cpu["theta_deg"].to_numpy()
        rcs_cpu   = sub_cpu["rcs_db"].to_numpy()
        theta_gpu = sub_gpu["theta_deg"].to_numpy()
        rcs_gpu   = sub_gpu["rcs_db"].to_numpy()

        m = compute_rcs_metrics_single(theta_cpu,
                                       rcs_cpu,
                                       theta_gpu,
                                       rcs_gpu,
                                       unwrap=UNWRAP_ANGLES)
        m["freq_ghz"] = f
        metrics_by_freq.append(m)
        freqs_used.append(f)

    freqs_used = np.array(freqs_used, dtype=float)

    # Compute global summary (across frequencies)
    maxAbs_all = np.array([m["maxAbsDiff_dB"] for m in metrics_by_freq])
    meanAbs_all = np.array([m["meanAbsDiff_dB"] for m in metrics_by_freq])
    corr_all = np.array([m["corr_coeff"] for m in metrics_by_freq])

    global_metrics = {
        "global_max_maxAbsDiff_dB": float(np.max(maxAbs_all)),
        "global_mean_maxAbsDiff_dB": float(np.mean(maxAbs_all)),
        "global_max_meanAbsDiff_dB": float(np.max(meanAbs_all)),
        "global_mean_corr_coeff": float(np.mean(corr_all)),
        "global_min_corr_coeff": float(np.min(corr_all)),
    }

    # Pick one representative frequency to show full plots
    # Option 1: worst-case by max |ΔRCS|
    worst_idx = int(np.argmax(maxAbs_all))
    worst_freq = metrics_by_freq[worst_idx]["freq_ghz"]
    worst_metrics = metrics_by_freq[worst_idx]

    print("-------------------------------------------------")
    print("Frequencies analyzed (GHz):")
    print(freqs_used)
    print("Global summary:")
    print(f"  Max of Max |ΔRCS|     : {global_metrics['global_max_maxAbsDiff_dB']:.3f} dB")
    print(f"  Mean of Max |ΔRCS|    : {global_metrics['global_mean_maxAbsDiff_dB']:.3f} dB")
    print(f"  Max of Mean |ΔRCS|    : {global_metrics['global_max_meanAbsDiff_dB']:.3f} dB")
    print(f"  Mean Corr. Coeff.     : {global_metrics['global_mean_corr_coeff']:.5f}")
    print(f"  Min Corr. Coeff.      : {global_metrics['global_min_corr_coeff']:.5f}")
    print("Worst-case frequency by Max |ΔRCS|: "
          f"{worst_freq:.3f} GHz (Max |ΔRCS| = {maxAbs_all[worst_idx]:.3f} dB)")
    print("-------------------------------------------------")

    # Generate PDF report
    with PdfPages(OUTPUT_PDF) as pdf:
        add_sweep_title_page(pdf, META, global_metrics, freqs_used)
        add_metrics_vs_freq_page(pdf, freqs_used, metrics_by_freq)
        # Example full plots at worst-case frequency
        add_overlay_plot(pdf,
                         worst_metrics["theta_deg"],
                         worst_metrics["rcs_cpu_dB"],
                         worst_metrics["rcs_gpu_dB"],
                         worst_freq)
        add_diff_plot(pdf,
                      worst_metrics["theta_deg"],
                      worst_metrics["diff_dB"],
                      worst_freq)
        add_hist_plot(pdf,
                      worst_metrics["diff_dB"],
                      worst_freq)

    print("RCS Sweep Validation Report generated:")
    print(f"  {OUTPUT_PDF}")
    print("-------------------------------------------------")


if __name__ == "__main__":
    main()
