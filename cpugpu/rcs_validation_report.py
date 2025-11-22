"""
RCS CPU vs GPU Validation Script
- Loads CPU & GPU RCS results from CSV
- Computes accuracy metrics
- Generates a multi-page PDF report

Requirements:
    pip install numpy matplotlib pandas

Data format (CSV assumed):
    theta_deg, rcs_db

You can adapt load_rcs_csv() to your own format.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import datetime


# ------------------------------------------------------------
# CONFIGURATION (EDIT THIS BLOCK FOR YOUR USE CASE)
# ------------------------------------------------------------
CPU_CSV_PATH = "rcs_cpu.csv"
GPU_CSV_PATH = "rcs_gpu.csv"
OUTPUT_PDF   = "rcs_validation_report.pdf"

META = {
    "project_name": "RCS Cone Validation",
    "target_description": "1 m cone, 0.5 m base, 0.1 m tip radius",
    "solver_cpu": "MoM/PO - CPU Baseline",
    "solver_gpu": "MoM/PO - GPU Accelerated",
    "cpu_hardware": "Xeon Gold (example)",
    "gpu_hardware": "NVIDIA A100 (example)",
    "frequency_ghz": 10.0,
    "polarization": "HH",
}

# Set to True if your theta wraps around and you want to unwrap it
UNWRAP_ANGLES = False


# ------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------
def load_rcs_csv(path):
    """
    Load RCS data from a CSV file.
    Expected columns: theta_deg, rcs_db

    Returns
    -------
    theta_deg : np.ndarray (1D)
    rcs_dB    : np.ndarray (1D)
    """
    df = pd.read_csv(path)
    # Try common column names, allow some flexibility
    possible_theta_cols = ["theta_deg", "theta", "angle_deg", "angle"]
    possible_rcs_cols   = ["rcs_db", "rcs_dB", "rcs", "RCS_dB"]

    theta_col = None
    rcs_col   = None

    for c in df.columns:
        if c in possible_theta_cols:
            theta_col = c
        if c in possible_rcs_cols:
            rcs_col = c

    if theta_col is None:
        raise ValueError(f"No theta column found in {path}. "
                         f"Expected one of {possible_theta_cols}")
    if rcs_col is None:
        raise ValueError(f"No RCS column found in {path}. "
                         f"Expected one of {possible_rcs_cols}")

    theta_deg = df[theta_col].to_numpy(dtype=float)
    rcs_dB    = df[rcs_col].to_numpy(dtype=float)

    return theta_deg, rcs_dB


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


def compute_rcs_metrics(theta_cpu_deg,
                        rcs_cpu_dB,
                        theta_gpu_deg,
                        rcs_gpu_dB,
                        unwrap=False):
    """
    Compute comparison metrics between CPU and GPU RCS curves.

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

    # Pattern correlation in dB-domain (can also be done in linear domain)
    # Use zero-mean vectors for correlation
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
def add_title_page(pdf, meta, metrics):
    """Create a title/summary page."""
    fig = plt.figure(figsize=(8.5, 11))
    fig.clf()
    text = []

    text.append(f"RCS Solver Validation Report")
    text.append("")
    text.append(f"Date: {datetime.date.today().isoformat()}")
    text.append("")
    text.append(f"Project: {meta.get('project_name', 'N/A')}")
    text.append(f"Target: {meta.get('target_description', 'N/A')}")
    text.append("")
    text.append(f"CPU Solver: {meta.get('solver_cpu', 'N/A')}")
    text.append(f"CPU Hardware: {meta.get('cpu_hardware', 'N/A')}")
    text.append("")
    text.append(f"GPU Solver: {meta.get('solver_gpu', 'N/A')}")
    text.append(f"GPU Hardware: {meta.get('gpu_hardware', 'N/A')}")
    text.append("")
    text.append(f"Frequency: {meta.get('frequency_ghz', 'N/A')} GHz")
    text.append(f"Polarization: {meta.get('polarization', 'N/A')}")
    text.append("")
    text.append("Summary of Key Metrics:")
    text.append(f"  Max |ΔRCS|          : {metrics['maxAbsDiff_dB']:.3f} dB")
    text.append(f"  Mean |ΔRCS|         : {metrics['meanAbsDiff_dB']:.3f} dB")
    text.append(f"  95th pct |ΔRCS|     : {metrics['p95AbsDiff_dB']:.3f} dB")
    text.append(f"  RMS diff (linear)   : {metrics['rmsDiff_lin']:.3e}")
    text.append(f"  NRMSE (linear)      : {metrics['nrmse_lin']:.3e}")
    text.append(f"  Pattern correlation : {metrics['corr_coeff']:.5f}")

    full_text = "\n".join(text)

    plt.text(0.05, 0.95, full_text,
             va="top", ha="left", fontsize=12, family="monospace")
    plt.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def add_metrics_page(pdf, metrics):
    """Detailed metrics page."""
    fig = plt.figure(figsize=(8.5, 11))
    fig.clf()

    text = []
    text.append("Detailed Accuracy Metrics")
    text.append("")
    text.append("Amplitude (dB-domain):")
    text.append(f"  Max |ΔRCS| (dB)        : {metrics['maxAbsDiff_dB']:.6f}")
    text.append(f"  Mean |ΔRCS| (dB)       : {metrics['meanAbsDiff_dB']:.6f}")
    text.append(f"  95th percentile |ΔRCS| : {metrics['p95AbsDiff_dB']:.6f}")
    text.append("")
    text.append("Linear-domain metrics:")
    text.append(f"  RMS difference (σ_lin) : {metrics['rmsDiff_lin']:.6e}")
    text.append(f"  NRMSE (σ_lin)          : {metrics['nrmse_lin']:.6e}")
    text.append("")
    text.append("Pattern correlation (dB-domain):")
    text.append(f"  Corr. coefficient      : {metrics['corr_coeff']:.6f}")
    text.append("")
    text.append("Interpretation guidelines (typical):")
    text.append("  Max |ΔRCS| < 0.2 dB  → Excellent match")
    text.append("  Max |ΔRCS| < 0.5 dB  → Good/acceptable")
    text.append("  Max |ΔRCS| > 1.0 dB  → Investigate (precision/tolerance issues)")
    text.append("")
    text.append("NRMSE (linear domain):")
    text.append("  < 1%   → Very high fidelity")
    text.append("  1–5%   → Acceptable for most PO/MoM RCS work")
    text.append("  > 10%  → Potential numerical or modeling issue")

    full_text = "\n".join(text)
    plt.text(0.05, 0.95, full_text,
             va="top", ha="left", fontsize=11, family="monospace")
    plt.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def add_overlay_plot(pdf, theta_deg, rcs_cpu_dB, rcs_gpu_dB):
    """CPU vs GPU RCS overlay plot."""
    fig = plt.figure(figsize=(8.5, 5))
    plt.plot(theta_deg, rcs_cpu_dB, label="CPU", linewidth=1.5)
    plt.plot(theta_deg, rcs_gpu_dB, "--", label="GPU", linewidth=1.5)
    plt.grid(True)
    plt.xlabel("Angle (deg)")
    plt.ylabel("RCS (dB)")
    plt.title("CPU vs GPU RCS Comparison")
    plt.legend()
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def add_diff_plot(pdf, theta_deg, diff_dB):
    """ΔRCS vs angle plot."""
    fig = plt.figure(figsize=(8.5, 5))
    plt.plot(theta_deg, diff_dB, linewidth=1.5)
    plt.grid(True)
    plt.xlabel("Angle (deg)")
    plt.ylabel("ΔRCS = GPU - CPU (dB)")
    plt.title("Difference Plot (ΔRCS vs Angle)")
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def add_hist_plot(pdf, diff_dB):
    """Histogram of ΔRCS."""
    fig = plt.figure(figsize=(8.5, 5))
    plt.hist(diff_dB, bins=40)
    plt.grid(True)
    plt.xlabel("ΔRCS (dB)")
    plt.ylabel("Count")
    plt.title("Histogram of ΔRCS (GPU - CPU)")
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ------------------------------------------------------------
# MAIN ENTRY POINT
# ------------------------------------------------------------
def main():
    # Load data
    theta_cpu_deg, rcs_cpu_dB = load_rcs_csv(CPU_CSV_PATH)
    theta_gpu_deg, rcs_gpu_dB = load_rcs_csv(GPU_CSV_PATH)

    # Compute metrics
    metrics = compute_rcs_metrics(theta_cpu_deg,
                                  rcs_cpu_dB,
                                  theta_gpu_deg,
                                  rcs_gpu_dB,
                                  unwrap=UNWRAP_ANGLES)

    # Generate PDF report
    with PdfPages(OUTPUT_PDF) as pdf:
        add_title_page(pdf, META, metrics)
        add_metrics_page(pdf, metrics)
        add_overlay_plot(pdf,
                         metrics["theta_deg"],
                         metrics["rcs_cpu_dB"],
                         metrics["rcs_gpu_dB"])
        add_diff_plot(pdf,
                      metrics["theta_deg"],
                      metrics["diff_dB"])
        add_hist_plot(pdf,
                      metrics["diff_dB"])

    print("-------------------------------------------------")
    print("RCS Validation Report generated:")
    print(f"  {OUTPUT_PDF}")
    print("Key metrics:")
    print(f"  Max |ΔRCS| (dB)       : {metrics['maxAbsDiff_dB']:.3f}")
    print(f"  Mean |ΔRCS| (dB)      : {metrics['meanAbsDiff_dB']:.3f}")
    print(f"  95th pct |ΔRCS| (dB)  : {metrics['p95AbsDiff_dB']:.3f}")
    print(f"  RMS diff (linear)     : {metrics['rmsDiff_lin']:.3e}")
    print(f"  NRMSE (linear)        : {metrics['nrmse_lin']:.3e}")
    print(f"  Pattern correlation   : {metrics['corr_coeff']:.5f}")
    print("-------------------------------------------------")


if __name__ == "__main__":
    main()
