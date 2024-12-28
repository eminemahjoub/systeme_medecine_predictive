import os
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
from utils.data_loader import DataLoader
from utils.data_processor import DataProcessor
from utils.model_evaluator import ModelEvaluator
import json

class MedicalPredictionSystem:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = Path('../models/medical_model.joblib')
        self.scaler_path = Path('../models/scaler.joblib')
        self.metadata_path = Path('../models/metadata.json')
        self.feature_names = None
        self.last_trained = None
        self.performance_metrics = {}
        
    def prepare_data(self, data):
        """Prepare and preprocess the input data."""
        if self.scaler is None:
            raise ValueError("Scaler not initialized!")
        return self.scaler.transform(data)
        
    def train_new_model(self):
        """Train a new model with the latest data."""
        # Load and preprocess data
        data_loader = DataLoader()
        data_processor = DataProcessor()
        
        df = data_loader.load_heart_disease_data()
        X, y = data_processor.preprocess_data(df)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model with optimized hyperparameters
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        evaluator = ModelEvaluator()
        self.performance_metrics = evaluator.evaluate_model(
            self.model, X_test_scaled, y_test
        )
        
        # Calculate feature importance
        feature_importance = dict(zip(
            self.feature_names,
            self.model.feature_importances_
        ))
        
        # Save model and metadata
        self.save_model(feature_importance)
        
    def save_model(self, feature_importance=None):
        """Save the trained model and associated metadata."""
        if not self.model_path.parent.exists():
            os.makedirs(self.model_path.parent)
            
        # Save model and scaler
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        
        # Save metadata
        metadata = {
            'last_trained': datetime.now().isoformat(),
            'feature_names': self.feature_names,
            'performance_metrics': self.performance_metrics,
            'feature_importance': feature_importance
        }
        
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f)
            
    def load_model(self):
        """Load a trained model and its metadata."""
        if not self.model_path.exists():
            raise FileNotFoundError("No trained model found!")
            
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        
        with open(self.metadata_path, 'r') as f:
            metadata = json.load(f)
            
        self.feature_names = metadata['feature_names']
        self.last_trained = metadata['last_trained']
        self.performance_metrics = metadata['performance_metrics']
            
    def predict(self, patient_data):
        """Make predictions for patient data."""
        if self.model is None:
            raise ValueError("Model not trained or loaded!")
        
        processed_data = self.prepare_data(patient_data)
        predictions = self.model.predict(processed_data)
        probabilities = self.model.predict_proba(processed_data)
        
        return predictions, probabilities
        
    def get_feature_importance(self):
        """Get feature importance scores."""
        if self.model is None:
            raise ValueError("Model not trained or loaded!")
            
        return dict(zip(
            self.feature_names,
            self.model.feature_importances_
        ))
        
    def get_performance_metrics(self):
        """Get model performance metrics."""
        return self.performance_metrics
        
    def get_last_trained_date(self):
        """Get the date when the model was last trained."""
        return self.last_trained

def main():
    # Initialize the prediction system
    predictor = MedicalPredictionSystem()
    
    try:
        # Try to load existing model
        print("Loading existing model...")
        predictor.load_model()
        print("Model loaded successfully!")
        
    except FileNotFoundError:
        # Train new model if none exists
        print("No existing model found. Training new model...")
        predictor.train_new_model()
        print("Model trained successfully!")
    
    # Print model information
    print("\nModel Performance Metrics:")
    for metric, value in predictor.get_performance_metrics().items():
        print(f"{metric}: {value}")
    
    print("\nFeature Importance:")
    for feature, importance in predictor.get_feature_importance().items():
        print(f"{feature}: {importance:.4f}")
    
    # Example prediction
    data_loader = DataLoader()
    df = data_loader.load_heart_disease_data()
    sample_patient = df.drop('target', axis=1).iloc[0].values.reshape(1, -1)
    
    prediction, probability = predictor.predict(sample_patient)
    
    print(f"\nSample Patient Prediction:")
    print(f"Risk of Disease: {'Yes' if prediction[0] == 1 else 'No'}")
    print(f"Confidence: {max(probability[0]) * 100:.2f}%")

if __name__ == "__main__":
    main()
