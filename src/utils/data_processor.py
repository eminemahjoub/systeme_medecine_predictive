import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer

class DataProcessor:
    def __init__(self):
        self.numerical_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = RobustScaler()
        
    def preprocess_data(self, df):
        """
        Preprocess the data with advanced feature engineering.
        """
        # Make a copy to avoid modifying original data
        df = df.copy()
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Feature engineering
        df = self._engineer_features(df)
        
        # Prepare features and target
        if 'target' in df.columns:
            X = df.drop('target', axis=1)
            y = df['target']
        else:
            X = df
            y = None
            
        return X, y
        
    def _handle_missing_values(self, df):
        """Handle missing values in the dataset."""
        # Identify numerical and categorical columns
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Handle numerical missing values
        if len(numerical_cols) > 0:
            df[numerical_cols] = self.numerical_imputer.fit_transform(df[numerical_cols])
            
        # Handle categorical missing values
        if len(categorical_cols) > 0:
            df[categorical_cols] = self.categorical_imputer.fit_transform(df[categorical_cols])
            
        return df
        
    def _engineer_features(self, df):
        """
        Create new features and transform existing ones.
        This is a placeholder - customize based on your specific dataset.
        """
        # Example feature engineering for medical data
        if 'mean radius' in df.columns and 'mean texture' in df.columns:
            # Create interaction features
            df['radius_texture_interaction'] = df['mean radius'] * df['mean texture']
            
            # Create polynomial features for important measurements
            df['radius_squared'] = df['mean radius'] ** 2
            df['texture_squared'] = df['mean texture'] ** 2
            
        # Add more feature engineering based on domain knowledge
        # For example, BMI calculation if height and weight are present
        if 'height' in df.columns and 'weight' in df.columns:
            df['bmi'] = df['weight'] / (df['height'] ** 2)
            
        return df
        
    def scale_features(self, X_train, X_test=None):
        """
        Scale the features using RobustScaler.
        RobustScaler is less influenced by outliers than StandardScaler.
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
            
        return X_train_scaled
