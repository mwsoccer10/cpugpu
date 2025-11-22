import numpy as np
import pandas as pd

print("Generating multi-frequency RCS test data...")

# Frequency sweep: 1 to 50 GHz, 10 points
frequencies = np.linspace(1, 50, 10)
theta = np.linspace(-90, 90, 181)

data_cpu = []
data_gpu = []

np.random.seed(42)

for freq in frequencies:
    # Simple frequency-dependent pattern
    # RCS varies with both angle and frequency
    wavelength_m = 0.3 / freq  # c/f in meters
    ka = 2 * np.pi / wavelength_m  # wave number * characteristic length (1m)
    
    for th in theta:
        # Simple cosine pattern modulated by frequency
        rcs_base = -10 + 20 * np.cos(np.deg2rad(th))**2 + 5 * np.sin(ka * np.deg2rad(th))
        
        # CPU (reference)
        data_cpu.append({
            'freq_ghz': freq,
            'theta_deg': th,
            'rcs_db': rcs_base
        })
        
        # GPU (with small noise)
        noise = 0.1 * np.random.randn()
        data_gpu.append({
            'freq_ghz': freq,
            'theta_deg': th,
            'rcs_db': rcs_base + noise
        })

df_cpu = pd.DataFrame(data_cpu)
df_gpu = pd.DataFrame(data_gpu)

df_cpu.to_csv('rcs_cpu_multi.csv', index=False)
df_gpu.to_csv('rcs_gpu_multi.csv', index=False)

print(f"✓ Created rcs_cpu_multi.csv ({len(df_cpu)} rows, {len(frequencies)} frequencies)")
print(f"✓ Created rcs_gpu_multi.csv ({len(df_gpu)} rows, {len(frequencies)} frequencies)")
print(f"  Frequency range: {frequencies[0]:.1f} - {frequencies[-1]:.1f} GHz")
print(f"  Angles per frequency: {len(theta)}")
