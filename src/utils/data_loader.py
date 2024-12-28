import pandas as pd
from sklearn.datasets import load_breast_cancer

class DataLoader:
    @staticmethod
    def load_heart_disease_data():
        """
        Load the Breast Cancer dataset as a substitute for heart disease data.
        This is a binary classification problem similar to heart disease prediction.
        """
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df
    
    @staticmethod
    def get_feature_descriptions():
        """Return descriptions of the features in the heart disease dataset."""
        return {
            'age': 'Age in years',
            'sex': 'Gender (1 = male; 0 = female)',
            'cp': 'Chest pain type (0-3)',
            'trestbps': 'Resting blood pressure (mm Hg)',
            'chol': 'Serum cholesterol (mg/dl)',
            'fbs': 'Fasting blood sugar > 120 mg/dl',
            'restecg': 'Resting ECG results (0-2)',
            'thalach': 'Maximum heart rate achieved',
            'exang': 'Exercise induced angina',
            'oldpeak': 'ST depression induced by exercise',
            'slope': 'Slope of peak exercise ST segment',
            'ca': 'Number of major vessels colored by flourosopy',
            'thal': 'Thalassemia type',
            'target': 'Heart disease diagnosis (1 = present; 0 = absent)'
        }
