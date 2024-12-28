import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score
)
from sklearn.model_selection import cross_val_score, learning_curve
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def evaluate_model(self, model, X_test, y_test):
        """Comprehensive model evaluation."""
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate basic metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_prob)
        
        # Add detailed classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics['detailed_report'] = report
        
        # Calculate cross-validation score
        cv_scores = cross_val_score(model, X_test, y_test, cv=5)
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        return metrics
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, y_prob=None):
        """Calculate various performance metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        
        if y_prob is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
                metrics['average_precision'] = average_precision_score(y_true, y_prob)
            except Exception as e:
                print(f"Warning: Could not calculate ROC-AUC or PR-AUC: {str(e)}")
                
        return metrics
        
    def plot_confusion_matrix(self, y_true, y_pred, labels=None):
        """Plot confusion matrix with enhanced styling."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        
        # Plot heatmap with improved styling
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            square=True,
            cbar_kws={'shrink': .8},
            annot_kws={'size': 12}
        )
        
        plt.title('Confusion Matrix', pad=20, size=14)
        plt.ylabel('True Label', size=12)
        plt.xlabel('Predicted Label', size=12)
        
        if labels:
            plt.xticks(np.arange(len(labels)) + 0.5, labels, rotation=45)
            plt.yticks(np.arange(len(labels)) + 0.5, labels, rotation=0)
            
        plt.tight_layout()
        return plt
        
    def plot_feature_importance(self, model, feature_names):
        """Plot feature importance with enhanced visualization."""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(12, 6))
            plt.title('Feature Importance', size=14, pad=20)
            
            # Create bar plot
            bars = plt.bar(range(len(importances)), 
                          importances[indices],
                          align='center',
                          color=plt.cm.Blues(np.linspace(0.3, 0.9, len(importances))))
            
            # Customize plot
            plt.xticks(range(len(importances)), 
                      [feature_names[i] for i in indices],
                      rotation=45,
                      ha='right')
            plt.xlabel('Features', size=12)
            plt.ylabel('Importance', size=12)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom')
                
            plt.tight_layout()
            return plt
        else:
            print("Model doesn't have feature importance attribute")
            return None
            
    def plot_learning_curve(self, model, X, y, cv=5):
        """Plot learning curve to analyze model performance vs training size."""
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y,
            cv=cv,
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy'
        )
        
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label='Training score', color='blue', marker='o')
        plt.plot(train_sizes, test_mean, label='Cross-validation score', color='red', marker='o')
        
        # Add bands for standard deviation
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='red')
        
        # Customize plot
        plt.title('Learning Curve', size=14, pad=20)
        plt.xlabel('Training Examples', size=12)
        plt.ylabel('Score', size=12)
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return plt
        
    def plot_precision_recall_curve(self, y_true, y_prob):
        """Plot precision-recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)
        
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, color='blue', marker='.')
        plt.fill_between(recall, precision, alpha=0.2, color='blue')
        
        plt.title('Precision-Recall Curve', size=14, pad=20)
        plt.xlabel('Recall', size=12)
        plt.ylabel('Precision', size=12)
        plt.text(0.6, 0.95, f'Average Precision = {avg_precision:.3f}')
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        return plt
