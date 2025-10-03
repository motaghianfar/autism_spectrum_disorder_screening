#!/usr/bin/env python3
"""
Script to regenerate figures from existing analysis results.
"""

import sys
import os
import pandas as pd
import pickle
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.visualization import Visualizer

def main():
    print("🔄 REGENERATING FIGURES")
    
    visualizer = Visualizer()
    os.makedirs('figures', exist_ok=True)
    
    # Load data (you might need to adjust paths)
    try:
        df = pd.read_csv('data/train.csv')
        print("✅ Data loaded successfully")
    except FileNotFoundError:
        print("❌ Data file not found")
        return
    
    # Generate figures
    visualizer.plot_asd_prevalence(df)
    print("✅ Figure 1: ASD Prevalence generated")
    
    # Note: For other figures, you would need to load saved results
    print("📝 Note: Other figures require saved model results")
    print("   Run scripts/run_analysis.py first for complete analysis")

if __name__ == "__main__":
    main()
