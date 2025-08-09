#!/usr/bin/env python3
"""
Test script to verify the updated notebook works with the current dataset
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_dataset_compatibility():
    """Test that the dataset matches notebook expectations"""
    try:
        # Load the dataset
        df = pd.read_csv('./data/housing_data.csv')
        
        print("Dataset Test Results:")
        print("=" * 40)
        print(f"✓ Dataset loaded successfully")
        print(f"✓ Shape: {df.shape}")
        print(f"✓ Columns: {list(df.columns)}")
        
        # Check required columns
        required_cols = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 
                        'mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                        'airconditioning', 'parking', 'prefarea', 'furnishingstatus']
        
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            print(f"✗ Missing columns: {missing_cols}")
            return False
        else:
            print("✓ All required columns present")
        
        # Check data types
        print(f"✓ Target variable 'price' type: {df['price'].dtype}")
        print(f"✓ No missing values: {df.isnull().sum().sum() == 0}")
        
        # Test basic statistics
        print(f"✓ Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
        print(f"✓ Average price: ${df['price'].mean():,.0f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_model_imports():
    """Test that required libraries can be imported"""
    try:
        print("\nLibrary Import Test:")
        print("=" * 40)
        
        import shap
        print(f"✓ SHAP v{shap.__version__}")
        
        import lime
        print(f"✓ LIME (lime)")
        
        from sklearn.ensemble import RandomForestRegressor
        print("✓ Scikit-learn models")
        
        import matplotlib.pyplot as plt
        print("✓ Matplotlib")
        
        import seaborn as sns
        print("✓ Seaborn")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

if __name__ == "__main__":
    print("Testing Notebook Compatibility")
    print("=" * 50)
    
    dataset_ok = test_dataset_compatibility()
    imports_ok = test_model_imports()
    
    if dataset_ok and imports_ok:
        print("\n🎉 All tests passed! Notebook should work correctly.")
    else:
        print("\n⚠️ Some tests failed. Please check the issues above.")