"""
Data Processing Module
Handles data loading, cleaning, and preprocessing operations
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.original_data = None
        self.processed_data = None
        self.feature_columns = None  # Store feature column names from training
        
    def load_data(self, file_path):
        """Load data from CSV file"""
        try:
            self.original_data = pd.read_csv(file_path)
            self.processed_data = self.original_data.copy()
            return self.original_data
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def get_data_info(self):
        """Get comprehensive data information"""
        if self.original_data is None:
            return None
            
        info = {
            'shape': self.original_data.shape,
            'columns': list(self.original_data.columns),
            'dtypes': self.original_data.dtypes.to_dict(),
            'missing_values': self.original_data.isnull().sum().to_dict(),
            'memory_usage': self.original_data.memory_usage(deep=True).sum(),
            'numeric_columns': list(self.original_data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.original_data.select_dtypes(include=['object']).columns)
        }
        
        return info
    
    def detect_outliers(self, column, method='iqr'):
        """Detect outliers in a numeric column"""
        if column not in self.original_data.columns:
            return None
            
        data = self.original_data[column].dropna()
        
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            
        elif method == 'z_score':
            z_scores = np.abs((data - data.mean()) / data.std())
            outliers = data[z_scores > 3]
            
        return {
            'outliers': outliers.tolist(),
            'outlier_indices': outliers.index.tolist(),
            'count': len(outliers),
            'percentage': (len(outliers) / len(data)) * 100
        }
    
    def handle_missing_values(self, strategy='mean', columns=None):
        """Handle missing values using specified strategy"""
        if columns is None:
            columns = self.processed_data.columns
            
        for column in columns:
            if column not in self.processed_data.columns:
                continue
                
            if self.processed_data[column].isnull().sum() == 0:
                continue
                
            if self.processed_data[column].dtype in ['object']:
                # Categorical columns
                if strategy == 'mode':
                    fill_value = self.processed_data[column].mode().iloc[0] if not self.processed_data[column].mode().empty else 'Unknown'
                else:
                    fill_value = 'Unknown'
                self.processed_data[column].fillna(fill_value, inplace=True)
            else:
                # Numeric columns
                if strategy == 'mean':
                    fill_value = self.processed_data[column].mean()
                elif strategy == 'median':
                    fill_value = self.processed_data[column].median()
                elif strategy == 'mode':
                    fill_value = self.processed_data[column].mode().iloc[0] if not self.processed_data[column].mode().empty else 0
                else:
                    fill_value = 0
                    
                self.processed_data[column].fillna(fill_value, inplace=True)
    
    def encode_categorical_variables(self, method='onehot', columns=None):
        """Encode categorical variables"""
        if columns is None:
            columns = self.processed_data.select_dtypes(include=['object']).columns
            
        encoded_data = self.processed_data.copy()
        
        for column in columns:
            if column not in encoded_data.columns:
                continue
                
            if method == 'onehot':
                # One-hot encoding
                dummies = pd.get_dummies(encoded_data[column], prefix=column)
                encoded_data = pd.concat([encoded_data, dummies], axis=1)
                encoded_data.drop(column, axis=1, inplace=True)
                
            elif method == 'label':
                # Label encoding
                le = LabelEncoder()
                encoded_data[column] = le.fit_transform(encoded_data[column].astype(str))
                self.encoders[column] = le
        
        return encoded_data
    
    def scale_features(self, method='standard', columns=None):
        """Scale numeric features"""
        if columns is None:
            columns = self.processed_data.select_dtypes(include=[np.number]).columns
            
        scaled_data = self.processed_data.copy()
        
        for column in columns:
            if column not in scaled_data.columns:
                continue
                
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            else:
                continue
                
            scaled_data[column] = scaler.fit_transform(scaled_data[[column]])
            self.scalers[column] = scaler
            
        return scaled_data
    
    def prepare_model_data(self, target_column, test_size=0.2, random_state=42):
        """Prepare data for machine learning models"""
        if target_column not in self.processed_data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # First handle missing values
        self.handle_missing_values(strategy='median')
            
        # Create a copy for processing
        df_processed = self.processed_data.copy()
        
        # Encode categorical variables using the same method as prediction
        # Convert yes/no to 1/0 for binary categorical variables
        binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
        
        for col in binary_cols:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].map({'yes': 1, 'no': 0})
        
        # One-hot encode furnishingstatus
        if 'furnishingstatus' in df_processed.columns:
            furnishing_dummies = pd.get_dummies(df_processed['furnishingstatus'], prefix='furnishing')
            df_processed = pd.concat([df_processed, furnishing_dummies], axis=1)
            df_processed.drop('furnishingstatus', axis=1, inplace=True)
            
            # Ensure furnishing dummy columns are numeric (int64)
            furnishing_cols = [col for col in df_processed.columns if col.startswith('furnishing_')]
            for col in furnishing_cols:
                df_processed[col] = df_processed[col].astype(int)
        
        # Separate features and target
        X = df_processed.drop(target_column, axis=1)
        y = df_processed[target_column]
        
        # Ensure only numeric columns
        X = X.select_dtypes(include=[np.number])
        
        # Handle any remaining NaN values
        X = X.fillna(X.median())
        
        # Store feature names for later use
        self.feature_columns = list(X.columns)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test
    
    def get_correlation_matrix(self):
        """Get correlation matrix for numeric columns"""
        numeric_data = self.processed_data.select_dtypes(include=[np.number])
        return numeric_data.corr()
    
    def reset_data(self):
        """Reset processed data to original"""
        if self.original_data is not None:
            self.processed_data = self.original_data.copy()
    
    def process_prediction_input(self, input_data):
        """Process input data for prediction (encode categorical variables)"""
        try:
            # Make a copy to avoid modifying the original
            processed_input = input_data.copy()
            
            # Encode categorical variables to match training data format
            # Convert yes/no to 1/0
            yes_no_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                            'airconditioning', 'prefarea']
            
            for col in yes_no_columns:
                if col in processed_input.columns:
                    processed_input[col] = processed_input[col].map({'yes': 1, 'no': 0})
            
            # One-hot encode furnishing status
            if 'furnishingstatus' in processed_input.columns:
                furnishing_dummies = pd.get_dummies(processed_input['furnishingstatus'], 
                                                  prefix='furnishing')
                
                # Ensure all expected columns are present
                expected_furnishing_cols = ['furnishing_furnished', 'furnishing_semi-furnished', 'furnishing_unfurnished']
                for col in expected_furnishing_cols:
                    if col not in furnishing_dummies.columns:
                        furnishing_dummies[col] = 0
                
                # Drop original column and add dummy columns
                processed_input = processed_input.drop('furnishingstatus', axis=1)
                processed_input = pd.concat([processed_input, furnishing_dummies[expected_furnishing_cols]], axis=1)
                
                # Ensure furnishing dummy columns are numeric (int64)
                for col in expected_furnishing_cols:
                    processed_input[col] = processed_input[col].astype(int)
            
            # If we have stored feature columns from training, use them
            if hasattr(self, 'feature_columns') and self.feature_columns:
                # Add missing columns with default values
                for col in self.feature_columns:
                    if col not in processed_input.columns:
                        processed_input[col] = 0
                
                # Reorder columns to match training data exactly
                processed_input = processed_input[self.feature_columns]
            else:
                # Fallback: use expected columns based on typical housing data structure
                expected_columns = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 
                                  'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 
                                  'parking', 'prefarea', 'furnishing_furnished', 
                                  'furnishing_semi-furnished', 'furnishing_unfurnished']
                
                # Add missing columns with default values
                for col in expected_columns:
                    if col not in processed_input.columns:
                        processed_input[col] = 0
                
                # Reorder columns to match expected structure
                available_cols = [col for col in expected_columns if col in processed_input.columns]
                processed_input = processed_input[available_cols]
            
            # Ensure all values are numeric
            processed_input = processed_input.select_dtypes(include=[np.number])
            
            # Handle any NaN values
            processed_input = processed_input.fillna(0)
            
            return processed_input
            
        except Exception as e:
            raise Exception(f"Error processing prediction input: {str(e)}")