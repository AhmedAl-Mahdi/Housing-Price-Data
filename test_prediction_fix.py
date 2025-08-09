#!/usr/bin/env python3
"""
Test script to verify that the prediction processing fix works correctly
"""

import sys
import os
import pandas as pd

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processing import DataProcessor
from model_evaluation import ModelEvaluator

def test_prediction_pipeline():
    """Test the complete prediction pipeline"""
    print("🧪 Testing Prediction Pipeline...")
    
    # Load data
    print("📊 Loading data...")
    processor = DataProcessor()
    data = processor.load_data('Housing_Price_Data.csv')
    print(f"✅ Loaded {len(data)} records")
    
    # Prepare model data
    print("🔧 Preparing model data...")
    X_train, X_test, y_train, y_test = processor.prepare_model_data(
        target_column='price',
        test_size=0.2,
        random_state=42
    )
    print(f"✅ Training features shape: {X_train.shape}")
    print(f"✅ Feature columns: {list(X_train.columns)}")
    
    # Train a simple model
    print("🤖 Training model...")
    evaluator = ModelEvaluator()
    results = evaluator.train_regression_models(X_train, X_test, y_train, y_test)
    
    # Get best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['metrics']['test_r2'])
    best_model = results[best_model_name]['model']
    print(f"✅ Best model: {best_model_name} (R² = {results[best_model_name]['metrics']['test_r2']:.4f})")
    
    # Test prediction input processing
    print("🔮 Testing prediction input...")
    
    # Create sample input
    sample_input = pd.DataFrame({
        'area': [7500],
        'bedrooms': [4],
        'bathrooms': [2],
        'stories': [2],
        'parking': [2],
        'mainroad': ['yes'],
        'guestroom': ['no'],
        'basement': ['yes'],
        'hotwaterheating': ['no'],
        'airconditioning': ['yes'],
        'prefarea': ['yes'],
        'furnishingstatus': ['furnished']
    })
    
    print(f"📝 Sample input:\n{sample_input}")
    
    # Process input
    processed_input = processor.process_prediction_input(sample_input)
    print(f"✅ Processed input shape: {processed_input.shape}")
    print(f"✅ Processed columns: {list(processed_input.columns)}")
    
    # Check if columns match
    training_cols = set(X_train.columns)
    prediction_cols = set(processed_input.columns)
    
    if training_cols == prediction_cols:
        print("✅ Column alignment: PERFECT MATCH!")
    else:
        print("❌ Column alignment: MISMATCH!")
        missing_in_prediction = training_cols - prediction_cols
        extra_in_prediction = prediction_cols - training_cols
        
        if missing_in_prediction:
            print(f"   Missing in prediction: {missing_in_prediction}")
        if extra_in_prediction:
            print(f"   Extra in prediction: {extra_in_prediction}")
    
    # Make prediction
    try:
        prediction = best_model.predict(processed_input)
        print(f"🎯 Prediction successful: ${prediction[0]:,.0f}")
        print("✅ Pipeline test PASSED!")
        return True
    except Exception as e:
        print(f"❌ Prediction failed: {str(e)}")
        print("❌ Pipeline test FAILED!")
        return False

if __name__ == "__main__":
    success = test_prediction_pipeline()
    if success:
        print("\n🎉 All tests passed! The prediction pipeline is working correctly.")
    else:
        print("\n💥 Tests failed! Please check the error messages above.")
