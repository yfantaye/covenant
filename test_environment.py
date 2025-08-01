#!/usr/bin/env python3
"""
Test script to verify the covenantv2 environment is working correctly.
"""

import sys
import os
import pandas as pd
import numpy as np

def test_environment():
    """Test that the environment is working correctly."""
    print("=== Covenant v2 Environment Test ===\n")
    
    # Test Python path
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Test key packages
    try:
        import yfinance as yf
        print("✅ yfinance imported successfully")
    except ImportError as e:
        print(f"❌ yfinance import failed: {e}")
    
    try:
        from sklearn.metrics import roc_auc_score
        print("✅ scikit-learn imported successfully")
    except ImportError as e:
        print(f"❌ scikit-learn import failed: {e}")
    
    try:
        import lightgbm as lgb
        print("✅ lightgbm imported successfully")
    except ImportError as e:
        print(f"❌ lightgbm import failed: {e}")
    
    try:
        import optuna
        print("✅ optuna imported successfully")
    except ImportError as e:
        print(f"❌ optuna import failed: {e}")
    
    # Test data manipulation
    try:
        df = pd.DataFrame({'test': [1, 2, 3]})
        print("✅ pandas working correctly")
    except Exception as e:
        print(f"❌ pandas test failed: {e}")
    
    # Test numpy
    try:
        arr = np.array([1, 2, 3])
        print("✅ numpy working correctly")
    except Exception as e:
        print(f"❌ numpy test failed: {e}")
    
    print("\n=== Environment Test Complete ===")

if __name__ == "__main__":
    test_environment() 