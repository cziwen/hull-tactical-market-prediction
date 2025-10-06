#!/usr/bin/env python3
"""
Quick test script to verify FeatureFactory setup.

Run this after creating the conda environment to ensure everything works.
"""

import sys
from pathlib import Path

print("=" * 70)
print("FeatureFactory Setup Test")
print("=" * 70)

# Test 1: Python version
print("\n1. Testing Python version...")
print(f"   Python {sys.version}")
if sys.version_info >= (3, 8):
    print("   ✓ Python version OK")
else:
    print("   ✗ Python 3.8+ required")

# Test 2: Required packages
print("\n2. Testing required packages...")
required_packages = [
    'pandas',
    'numpy',
    'scipy',
    'matplotlib',
    'seaborn',
    'yaml',
    'polars',
]

for package in required_packages:
    try:
        __import__(package)
        print(f"   ✓ {package}")
    except ImportError:
        print(f"   ✗ {package} - NOT INSTALLED")

# Test 3: Qlib (optional for basic functionality)
print("\n3. Testing Qlib (optional)...")
try:
    import qlib
    print(f"   ✓ Qlib version {qlib.__version__}")
except ImportError:
    print("   ⚠ Qlib not installed (optional)")
    print("     Install with: pip install pyqlib")

# Test 4: Project structure
print("\n4. Testing project structure...")
required_dirs = [
    'data/raw',
    'data/processed',
    'data/qlib_format',
    'configs',
    'notebooks',
    'src',
    'outputs/features',
    'outputs/reports',
]

for dir_path in required_dirs:
    full_path = Path(__file__).parent / dir_path
    if full_path.exists():
        print(f"   ✓ {dir_path}")
    else:
        print(f"   ✗ {dir_path} - MISSING")

# Test 5: Import custom modules
print("\n5. Testing custom modules...")
sys.path.insert(0, str(Path(__file__).parent / 'src'))

modules = [
    'data_converter',
    'factor_generator',
    'feature_evaluator',
    'custom_datahandler',
]

for module in modules:
    try:
        __import__(module)
        print(f"   ✓ {module}")
    except ImportError as e:
        print(f"   ✗ {module} - ERROR: {e}")

# Test 6: Data files
print("\n6. Testing data files...")
data_files = [
    'data/raw/train.csv',
    'data/raw/test.csv',
]

for file_path in data_files:
    full_path = Path(__file__).parent / file_path
    if full_path.exists():
        size_mb = full_path.stat().st_size / 1024 / 1024
        print(f"   ✓ {file_path} ({size_mb:.2f} MB)")
    else:
        print(f"   ⚠ {file_path} - NOT FOUND")

# Test 7: Config files
print("\n7. Testing config files...")
config_files = [
    'configs/data_config.yaml',
    'configs/factor_config.yaml',
]

for file_path in config_files:
    full_path = Path(__file__).parent / file_path
    if full_path.exists():
        print(f"   ✓ {file_path}")
    else:
        print(f"   ✗ {file_path} - MISSING")

# Test 8: Quick functionality test
print("\n8. Testing basic functionality...")
try:
    from factor_generator import FactorGenerator

    gen = FactorGenerator()
    lag_factors = gen.generate_lag_factors(features=['M1'], lags=[1, 5])

    if len(lag_factors) > 0:
        print(f"   ✓ Factor generation works ({len(lag_factors)} lag factors created)")
        print(f"     Example: {lag_factors[0]}")
    else:
        print("   ✗ Factor generation failed")

except Exception as e:
    print(f"   ✗ Error: {e}")

# Summary
print("\n" + "=" * 70)
print("Setup Test Complete!")
print("=" * 70)
print("\nNext steps:")
print("1. If Qlib is not installed: pip install pyqlib")
print("2. Run: jupyter notebook notebooks/01_data_preparation.ipynb")
print("3. Then: jupyter notebook notebooks/02_factor_exploration.ipynb")
print("=" * 70)
