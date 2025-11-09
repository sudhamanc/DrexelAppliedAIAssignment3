#!/usr/bin/env python3
"""
Fraud Detection Demo Runner
============================
This script runs both the binary and multiclass fraud detection models sequentially.

Usage:
    python rundemo.py

Features:
- Runs binary classification (XGBoost) first
- Then runs multiclass classification (MLP)
- Displays progress and timing information
- All visualizations saved to fraud_results/

Total estimated runtime: 30-40 minutes
- Binary classification: ~25-30 minutes
- Multiclass classification: ~5-8 minutes

Author: Sudhaman Chandrasekaran
Course: Applied AI - Fall 2025
"""

import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path


DATA_DIR = Path("data")
DATA_FILE = DATA_DIR / "creditcardFraudTransactions.csv"
DATA_ARCHIVE = DATA_DIR / "creditcardFraudTransactions.csv.zip"


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")


def run_script(script_name, description, estimated_time):
    """
    Run a Python script and track execution time
    
    Args:
        script_name: Name of the Python script to run
        description: Description of what the script does
        estimated_time: Estimated runtime as a string
    
    Returns:
        Tuple of (success: bool, elapsed_time: float)
    """
    print_header(f"{description}")
    print(f"Script: {script_name}")
    print(f"Estimated time: {estimated_time}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    start_time = time.time()
    
    try:
        # Run the script and capture output in real-time
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            text=True,
            capture_output=False  # Show output in real-time
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✓ {description} completed successfully!")
        print(f"Actual runtime: {elapsed_time / 60:.2f} minutes")
        
        return True, elapsed_time
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        
        print(f"\n✗ {description} failed!")
        print(f"Error code: {e.returncode}")
        print(f"Runtime before failure: {elapsed_time / 60:.2f} minutes")
        
        return False, elapsed_time
    
    except KeyboardInterrupt:
        elapsed_time = time.time() - start_time
        
        print(f"\n⚠ {description} interrupted by user!")
        print(f"Runtime before interruption: {elapsed_time / 60:.2f} minutes")
        
        return False, elapsed_time


def extract_packaged_dataset():
    """Extract the bundled dataset archive if available."""
    if not DATA_ARCHIVE.exists():
        return False

    print_header("DATASET SETUP")
    print("Found packaged dataset archive. Extracting now...")

    try:
        with zipfile.ZipFile(DATA_ARCHIVE) as archive:
            archive.extractall(DATA_DIR)
    except zipfile.BadZipFile:
        print("✗ Packaged archive appears to be corrupted. Delete it and rerun the demo.")
        return False
    except OSError as exc:
        print(f"✗ Unable to extract archive: {exc}")
        return False

    if DATA_FILE.exists():
        print(f"✓ Dataset extracted to {DATA_FILE}")
        return True

    print("✗ Archive extraction completed but dataset file is still missing.")
    return False


def ensure_dataset():
    """Ensure the fraud dataset is available locally."""
    if DATA_FILE.exists():
        print(f"✓ Dataset already present at {DATA_FILE}")
        return True

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if extract_packaged_dataset():
        return True

    print_header("DATASET SETUP")
    print("Packaged dataset not found. Please add `creditcardFraudTransactions.csv.zip`"
          " to the `data/` directory and rerun the demo.")
    return False


def main():
    """Main execution function"""
    if not ensure_dataset():
        print("\nDemo cannot continue without the dataset.")
        return 1

    print_header("FRAUD DETECTION DEMO - FULL PIPELINE")
    print("This script will run both fraud detection models:")
    print("  1. Binary Classification (XGBoost) - Advanced 7-strategy optimization")
    print("  2. Multiclass Classification (MLP) - 4-class risk level detection")
    print("\nTotal estimated time: 30-40 minutes")
    print("\nPress Ctrl+C at any time to stop. Starting immediately...\n")
    
    overall_start = time.time()
    results = []
    
    # Script configurations
    scripts = [
        {
            'name': 'fraud_binary_xgboost.py',
            'description': 'Binary Classification (XGBoost)',
            'estimated_time': '25-30 minutes'
        },
        {
            'name': 'fraud_multiclass_mlp.py',
            'description': 'Multiclass Classification (MLP)',
            'estimated_time': '5-8 minutes'
        }
    ]
    
    # Run each script
    for i, script in enumerate(scripts, 1):
        print(f"\n[{i}/{len(scripts)}] Starting {script['description']}...")
        
        success, elapsed = run_script(
            script['name'],
            script['description'],
            script['estimated_time']
        )
        
        results.append({
            'script': script['name'],
            'description': script['description'],
            'success': success,
            'time': elapsed
        })
        
        # If a script fails, stop the remaining runs (no interactive prompt)
        if not success:
            print("\n⚠ Aborting remaining scripts because of the failure above.")
            break
    
    # Print summary
    total_time = time.time() - overall_start
    
    print_header("DEMO COMPLETE - SUMMARY")
    print(f"Total runtime: {total_time / 60:.2f} minutes\n")
    
    print("Results:")
    print("-" * 80)
    for result in results:
        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        print(f"{status:12} | {result['description']:35} | {result['time'] / 60:6.2f} min")
    print("-" * 80)
    
    # Count successes
    successful = sum(1 for r in results if r['success'])
    total = len(results)
    
    print(f"\nCompleted: {successful}/{total} scripts successful")
    
    if successful == total:
        print("\n🎉 All models executed successfully!")
        print("\nGenerated visualizations can be found in: fraud_results/")
        print("  - fraud_binary_confusion_matrix.png")
        print("  - fraud_binary_metrics.png")
        print("  - fraud_binary_roc_curve.png")
        print("  - multiclass_confusion_matrix.png")
        print("  - multiclass_metrics.png")
        print("  - multiclass_per_class.png")
        print("\nSee README.md for detailed results and analysis.")
        return 0
    else:
        print("\n⚠ Some scripts failed. Please check the error messages above.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠ Demo interrupted by user. Exiting...")
        sys.exit(130)  # Standard exit code for SIGINT
