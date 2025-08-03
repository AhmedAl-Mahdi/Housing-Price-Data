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
    
    # Location features
    neighborhoods = ['Downtown', 'Suburban', 'Uptown', 'Riverside', 'Hillside', 'Industrial', 'University']
    data['Neighborhood'] = np.random.choice(neighborhoods, n_samples)
    
    # Property features
    data['Bedrooms'] = np.random.choice([1, 2, 3, 4, 5, 6], n_samples, p=[0.1, 0.25, 0.35, 0.2, 0.08, 0.02])
    data['Bathrooms'] = data['Bedrooms'] + np.random.choice([-1, 0, 1, 2], n_samples, p=[0.1, 0.4, 0.4, 0.1])
    data['Bathrooms'] = np.clip(data['Bathrooms'], 1, 6)
    
    # Square footage - correlated with bedrooms
    base_sqft = data['Bedrooms'] * 400 + np.random.normal(500, 200, n_samples)
    data['Square_Feet'] = np.clip(base_sqft, 500, 5000).astype(int)
    
    # Age of house
    data['Year_Built'] = np.random.randint(1950, 2024, n_samples)
    data['Age'] = 2024 - data['Year_Built']
    
    # Property type
    property_types = ['Single Family', 'Townhouse', 'Condo', 'Duplex']
    data['Property_Type'] = np.random.choice(property_types, n_samples, p=[0.6, 0.2, 0.15, 0.05])
    
    # Additional features
    data['Garage_Spaces'] = np.random.choice([0, 1, 2, 3], n_samples, p=[0.2, 0.3, 0.4, 0.1])
    data['Has_Pool'] = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    data['Has_Fireplace'] = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    data['Has_AC'] = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    
    # Lot size
    data['Lot_Size'] = np.random.normal(8000, 3000, n_samples)
    data['Lot_Size'] = np.clip(data['Lot_Size'], 2000, 20000).astype(int)
    
    # Distance to amenities (in miles)
    data['Distance_to_School'] = np.random.exponential(1.5, n_samples)
    data['Distance_to_Shopping'] = np.random.exponential(2.0, n_samples)
    data['Distance_to_Highway'] = np.random.exponential(3.0, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Calculate price based on features (with some noise)
    price_base = (
        df['Square_Feet'] * 120 +  # $120 per sq ft
        df['Bedrooms'] * 15000 +   # $15k per bedroom
        df['Bathrooms'] * 8000 +   # $8k per bathroom
        df['Garage_Spaces'] * 5000 + # $5k per garage space
        df['Has_Pool'] * 25000 +   # $25k for pool
        df['Has_Fireplace'] * 8000 + # $8k for fireplace
        df['Has_AC'] * 5000 +      # $5k for AC
        df['Lot_Size'] * 5 +       # $5 per sq ft of lot
        -df['Age'] * 1000          # Depreciation
    )
    
    # Neighborhood multipliers
    neighborhood_multipliers = {
        'Downtown': 1.3,
        'Uptown': 1.2,
        'Riverside': 1.1,
        'Hillside': 1.15,
        'Suburban': 1.0,
        'University': 0.9,
        'Industrial': 0.8
    }
    
    # Apply neighborhood effects
    for i, neighborhood in enumerate(df['Neighborhood']):
        price_base[i] *= neighborhood_multipliers[neighborhood]
    
    # Add distance penalties
    price_base -= df['Distance_to_School'] * 3000
    price_base -= df['Distance_to_Shopping'] * 2000
    price_base -= df['Distance_to_Highway'] * 1000
    
    # Add random noise
    noise = np.random.normal(0, 20000, n_samples)
    df['Price'] = price_base + noise
    
    # Ensure positive prices
    df['Price'] = np.clip(df['Price'], 50000, 2000000).astype(int)
    
    # Add some missing values to simulate real data
    missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
    df.loc[missing_indices[:len(missing_indices)//2], 'Lot_Size'] = np.nan
    df.loc[missing_indices[len(missing_indices)//2:], 'Year_Built'] = np.nan
    
    # Round distance features
    df['Distance_to_School'] = df['Distance_to_School'].round(1)
    df['Distance_to_Shopping'] = df['Distance_to_Shopping'].round(1)
    df['Distance_to_Highway'] = df['Distance_to_Highway'].round(1)
    
    return df

if __name__ == "__main__":
    # Generate dataset
    print("Generating housing price dataset...")
    housing_data = generate_housing_data(2000)
    
    # Save to CSV
    housing_data.to_csv('data/housing_data.csv', index=False)
    print(f"Dataset saved to data/housing_data.csv")
    print(f"Dataset shape: {housing_data.shape}")
    print("\nDataset info:")
    print(housing_data.info())
    print("\nFirst few rows:")
    print(housing_data.head())
    print(f"\nPrice statistics:")
    print(housing_data['Price'].describe())