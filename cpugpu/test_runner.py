#!/usr/bin/env python
import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)

try:
    import numpy
    print("✓ numpy imported")
except Exception as e:
    print("✗ numpy error:", e)

try:
    import pandas
    print("✓ pandas imported")
except Exception as e:
    print("✗ pandas error:", e)

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-GUI backend
    print("✓ matplotlib imported (Agg backend)")
except Exception as e:
    print("✗ matplotlib error:", e)

print("\nAttempting to run validation script...")
try:
    exec(open('rcs_validation_report.py').read())
    print("✓ Script executed successfully")
except FileNotFoundError as e:
    print(f"✗ File not found: {e}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
