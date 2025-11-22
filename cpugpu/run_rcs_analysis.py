import numpy as np
import matplotlib.pyplot as plt
from compare_rcs_cpu_gpu import compare_rcs_cpu_gpu

# Synthetic test data
theta_cpu = np.linspace(-90, 90, 721)
rcs_true = -10 + 20 * np.cos(np.deg2rad(theta_cpu))**2
rcs_cpu = rcs_true

theta_gpu = np.linspace(-90, 90, 601)
rcs_true_gpu = np.interp(theta_gpu, theta_cpu, rcs_true)
rcs_gpu = rcs_true_gpu + 0.2 * np.random.randn(theta_gpu.size)  # ~0.2 dB noise

opts = {
    'title': 'Test Pattern (Single Freq)',
    'interp_method': 'linear',
    'unwrap': False,
}

stats = compare_rcs_cpu_gpu(theta_cpu, rcs_cpu,
                            theta_gpu, rcs_gpu,
                            opts)

# Save all figures
figs = [plt.figure(i) for i in plt.get_fignums()]
figs[0].savefig('rcs_overlay.png', dpi=150, bbox_inches='tight')
figs[1].savefig('rcs_difference.png', dpi=150, bbox_inches='tight')
figs[2].savefig('rcs_histogram.png', dpi=150, bbox_inches='tight')
print('\nPlots saved: rcs_overlay.png, rcs_difference.png, rcs_histogram.png')

plt.show()
