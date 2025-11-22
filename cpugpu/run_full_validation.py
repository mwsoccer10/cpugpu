import numpy as np
import pandas as pd
import os

print("Creating test CSV files...")

# Generate test data
theta = np.linspace(-90, 90, 361)
rcs_cpu = -10 + 20 * np.cos(np.deg2rad(theta))**2

# CPU data
df_cpu = pd.DataFrame({
    'theta_deg': theta,
    'rcs_db': rcs_cpu
})
df_cpu.to_csv('rcs_cpu.csv', index=False)
print(f"✓ Created rcs_cpu.csv ({len(df_cpu)} rows)")

# GPU data (with small noise)
np.random.seed(42)
rcs_gpu = rcs_cpu + 0.15 * np.random.randn(len(theta))
df_gpu = pd.DataFrame({
    'theta_deg': theta,
    'rcs_db': rcs_gpu
})
df_gpu.to_csv('rcs_gpu.csv', index=False)
print(f"✓ Created rcs_gpu.csv ({len(df_gpu)} rows)")

# Verify files exist
if os.path.exists('rcs_cpu.csv') and os.path.exists('rcs_gpu.csv'):
    print("\n✓ CSV files verified")
    print("\nRunning validation report...")
    
    # Import and run the validation script
    import rcs_validation_report
    rcs_validation_report.main()
else:
    print("\n✗ CSV files not found!")
