import numpy as np
import pandas as pd

# Generate test data
theta = np.linspace(-90, 90, 361)
rcs_cpu = -10 + 20 * np.cos(np.deg2rad(theta))**2

# CPU data
pd.DataFrame({
    'theta_deg': theta,
    'rcs_db': rcs_cpu
}).to_csv('rcs_cpu.csv', index=False)

# GPU data (with small noise)
np.random.seed(42)
rcs_gpu = rcs_cpu + 0.15 * np.random.randn(len(theta))
pd.DataFrame({
    'theta_deg': theta,
    'rcs_db': rcs_gpu
}).to_csv('rcs_gpu.csv', index=False)

print('Created rcs_cpu.csv and rcs_gpu.csv')
