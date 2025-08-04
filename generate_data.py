"""
Housing Price Dataset Generator
Creates a synthetic housing dataset similar to real-world housing data
"""
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_housing_data(n_samples=2000):
    """Generate synthetic housing price data"""
    
    # Base features
    data = {}
    
    # Property features
    data['bedrooms'] = np.random.choice([1, 2, 3, 4, 5, 6], n_samples, p=[0.1, 0.25, 0.35, 0.2, 0.08, 0.02])
    data['bathrooms'] = data['bedrooms'] + np.random.choice([-1, 0, 1, 2], n_samples, p=[0.1, 0.4, 0.4, 0.1])
    data['bathrooms'] = np.clip(data['bathrooms'], 1, 6)
    
    # Area - correlated with bedrooms
    base_area = data['bedrooms'] * 400 + np.random.normal(500, 200, n_samples)
    data['area'] = np.clip(base_area, 500, 5000).astype(int)
    
    # Stories
    data['stories'] = np.random.choice([1, 2, 3, 4], n_samples, p=[0.3, 0.4, 0.25, 0.05])
    
    # Binary features (yes/no)
    data['mainroad'] = np.random.choice(['yes', 'no'], n_samples, p=[0.8, 0.2])
    data['guestroom'] = np.random.choice(['yes', 'no'], n_samples, p=[0.3, 0.7])
    data['basement'] = np.random.choice(['yes', 'no'], n_samples, p=[0.4, 0.6])
    data['hotwaterheating'] = np.random.choice(['yes', 'no'], n_samples, p=[0.2, 0.8])
    data['airconditioning'] = np.random.choice(['yes', 'no'], n_samples, p=[0.7, 0.3])
    data['prefarea'] = np.random.choice(['yes', 'no'], n_samples, p=[0.6, 0.4])
    
    # Parking spaces
    data['parking'] = np.random.choice([0, 1, 2, 3], n_samples, p=[0.2, 0.3, 0.4, 0.1])
    
    # Furnishing status
    data['furnishingstatus'] = np.random.choice(['furnished', 'semi-furnished', 'unfurnished'], 
                                               n_samples, p=[0.4, 0.3, 0.3])
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Calculate price based on features (with some noise)
    price_base = (
        df['area'] * 120 +  # $120 per sq ft
        df['bedrooms'] * 15000 +   # $15k per bedroom
        df['bathrooms'] * 8000 +   # $8k per bathroom
        df['stories'] * 5000 +   # $5k per story
        df['parking'] * 5000 +   # $5k per parking space
        (df['mainroad'] == 'yes') * 10000 +   # $10k for main road
        (df['guestroom'] == 'yes') * 8000 +   # $8k for guest room
        (df['basement'] == 'yes') * 12000 +   # $12k for basement
        (df['hotwaterheating'] == 'yes') * 3000 +   # $3k for hot water heating
        (df['airconditioning'] == 'yes') * 5000 +   # $5k for AC
        (df['prefarea'] == 'yes') * 15000   # $15k for preferred area
    )
    
    # Furnishing status adjustments
    furnishing_multipliers = {
        'furnished': 1.15,
        'semi-furnished': 1.05,
        'unfurnished': 1.0
    }
    
    # Apply furnishing effects
    for i, furnishing in enumerate(df['furnishingstatus']):
        price_base[i] *= furnishing_multipliers[furnishing]
    
    # Add random noise
    noise = np.random.normal(0, 20000, n_samples)
    df['price'] = price_base + noise
    
    # Ensure positive prices
    df['price'] = np.clip(df['price'], 50000, 20000000).astype(int)
    
    return df

if __name__ == "__main__":
    # Generate dataset
    print("Generating housing price dataset...")
    housing_data = generate_housing_data(2000)
    
    # Save to CSV
    housing_data.to_csv('Housing_Price_Data.csv', index=False)
    print(f"Dataset saved to Housing_Price_Data.csv")
    print(f"Dataset shape: {housing_data.shape}")
    print("\nDataset info:")
    print(housing_data.info())
    print("\nFirst few rows:")
    print(housing_data.head())
    print(f"\nPrice statistics:")
    print(housing_data['price'].describe())