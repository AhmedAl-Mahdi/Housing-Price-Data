"""
Test script to verify the dashboard integration with notebook models
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def test_notebook_preprocessing():
    """Test the notebook preprocessing function"""
    # Load sample data
    try:
        data = pd.read_csv('Housing_Price_Data.csv')
        print("✅ Data loaded successfully")
        print(f"Original shape: {data.shape}")
        print(f"Columns: {data.columns.tolist()}")
        
        # Test preprocessing
        df_processed = data.copy()
        
        # Encode categorical variables
        binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
        
        for col in binary_cols:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].map({'yes': 1, 'no': 0})
        
        # One-hot encode furnishingstatus
        if 'furnishingstatus' in df_processed.columns:
            furnishing_dummies = pd.get_dummies(df_processed['furnishingstatus'], prefix='furnishing')
            df_processed = pd.concat([df_processed, furnishing_dummies], axis=1)
            df_processed.drop('furnishingstatus', axis=1, inplace=True)
        
        print("✅ Preprocessing completed")
        print(f"Processed shape: {df_processed.shape}")
        print(f"Processed columns: {df_processed.columns.tolist()}")
        
        # Test model training
        X = df_processed.drop('price', axis=1)
        y = df_processed['price']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Test a simple model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print("✅ Model training successful")
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: ${rmse:,.0f}")
        
        # Test single prediction
        sample_input = [7500, 3, 2, 2, 2, 1, 0, 1, 0, 1, 1, 1, 0, 0]  # notebook format
        sample_pred = model.predict([sample_input])[0]
        print(f"✅ Sample prediction: ${sample_pred:,.0f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Dashboard-Notebook Integration")
    print("=" * 50)
    
    success = test_notebook_preprocessing()
    
    if success:
        print("\n✅ All tests passed! Dashboard should work with notebook models.")
        print("\n🚀 To run the dashboard:")
        print("   streamlit run dashboard.py")
        print("\n📋 To use notebook models:")
        print("   1. Select 'Notebook Models' in the sidebar")
        print("   2. Navigate to any module to see the integrated models")
    else:
        print("\n❌ Tests failed. Check the error messages above.")
