#!/usr/bin/env python3
"""
Simple test to verify model selection logic
"""

import sys
import os
import pandas as pd

# Add src directory to path
sys.path.append('src')

from data_processing import DataProcessor
from model_evaluation import ModelEvaluator

def test_model_selection():
    print("🧪 Testing Model Selection Logic...")
    
    # Load and process data
    processor = DataProcessor()
    data = processor.load_data('Housing_Price_Data.csv')
    X_train, X_test, y_train, y_test = processor.prepare_model_data('price')
    
    # Train models
    evaluator = ModelEvaluator()
    results = evaluator.train_regression_models(X_train, X_test, y_train, y_test)
    
    print("\n📊 Model Performance Comparison:")
    print("="*50)
    
    for name, result in results.items():
        metrics = result['metrics']
        print(f"\n{name}:")
        print(f"  Train R²: {metrics['train_r2']:.4f}")
        print(f"  Test R²:  {metrics['test_r2']:.4f}")
        print(f"  Test RMSE: ${metrics['test_rmse']:,.0f}")
    
    # Test selection logic
    print("\n🎯 Model Selection:")
    print("="*30)
    
    # Method 1: Using max with key
    best_model_1 = max(results.keys(), key=lambda x: results[x]['metrics']['test_r2'])
    best_r2_1 = results[best_model_1]['metrics']['test_r2']
    print(f"Method 1 - max() with key: {best_model_1} (R²={best_r2_1:.4f})")
    
    # Method 2: Manual comparison
    best_model_2 = None
    best_r2_2 = -1
    for name, result in results.items():
        test_r2 = result['metrics']['test_r2']
        if test_r2 > best_r2_2:
            best_r2_2 = test_r2
            best_model_2 = name
    
    print(f"Method 2 - manual loop: {best_model_2} (R²={best_r2_2:.4f})")
    
    # Method 3: Sorted list
    sorted_models = sorted(results.items(), key=lambda x: x[1]['metrics']['test_r2'], reverse=True)
    best_model_3 = sorted_models[0][0]
    best_r2_3 = sorted_models[0][1]['metrics']['test_r2']
    print(f"Method 3 - sorted list: {best_model_3} (R²={best_r2_3:.4f})")
    
    # Verify consistency
    if best_model_1 == best_model_2 == best_model_3:
        print(f"\n✅ All methods agree: {best_model_1} is the best model")
        return True
    else:
        print(f"\n❌ Methods disagree!")
        print(f"  Method 1: {best_model_1}")
        print(f"  Method 2: {best_model_2}")
        print(f"  Method 3: {best_model_3}")
        return False

if __name__ == "__main__":
    test_model_selection()
