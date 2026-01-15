#!/usr/bin/env python
"""
Simple batch runner for sentence_level_error.py
Analyzes multiple languages and configurations in one go
"""

import subprocess
import sys
import os
from datetime import datetime

# Configuration
LANGUAGES = ["am", "ha", "sw", "yo", "zu"]
CONFIGS = ["health", "tech"]
MAX_SAMPLES = 100
OUTPUT_DIR = "batch_analysis_results"

def run_analysis(config, lang, max_samples):
    """Run analysis for a single config/language pair"""
    print(f"\n{'='*60}")
    print(f"Analyzing: {config}/{lang}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable,
        "sentence_level_error.py",
        "--task", "mt",
        "--config", config,
        "--lang", lang,
        "--max-samples", str(max_samples),
        "--output-dir", os.path.join(OUTPUT_DIR, config)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Success: {config}/{lang}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {config}/{lang}")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Main batch runner"""
    print("="*60)
    print("BATCH QUALITY ANALYSIS")
    print("="*60)
    print(f"Configurations: {CONFIGS}")
    print(f"Languages: {LANGUAGES}")
    print(f"Max samples per run: {MAX_SAMPLES}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Track results
    total = len(CONFIGS) * len(LANGUAGES)
    success = 0
    failed = 0
    
    start_time = datetime.now()
    
    # Run analyses
    for config in CONFIGS:
        for lang in LANGUAGES:
            if run_analysis(config, lang, MAX_SAMPLES):
                success += 1
            else:
                failed += 1
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "="*60)
    print("BATCH ANALYSIS COMPLETE")
    print("="*60)
    print(f"Total runs: {total}")
    print(f"Successful: {success}")
    print(f"Failed: {failed}")
    print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print("="*60)

if __name__ == "__main__":
    main()
