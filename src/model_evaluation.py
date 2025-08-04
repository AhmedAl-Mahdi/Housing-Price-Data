"""
Model Evaluation Module
Contains functions for model training, evaluation, and visualization
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    def __init__(self):
        self.models = {}
        self.predictions = {}
        self.scores = {}
        
    def train_regression_models(self, X_train, X_test, y_train, y_test):
        """Train and evaluate regression models"""
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),  # Prevent overfitting
            'Random Forest (Deep)': RandomForestRegressor(n_estimators=100, random_state=42),  # Full depth
        }
        
        results = {}
        
        for name, model in models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_mse = mean_squared_error(y_train, train_pred)
            test_mse = mean_squared_error(y_test, test_pred)
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            train_mae = mean_absolute_error(y_train, train_pred)
            test_mae = mean_absolute_error(y_test, test_pred)
            
            results[name] = {
                'model': model,
                'train_predictions': train_pred,
                'test_predictions': test_pred,
                'metrics': {
                    'train_mse': train_mse,
                    'test_mse': test_mse,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'train_rmse': np.sqrt(train_mse),
                    'test_rmse': np.sqrt(test_mse)
                }
            }
            
            # Store for later use
            self.models[name] = model
            self.predictions[name] = {'train': train_pred, 'test': test_pred}
            self.scores[name] = results[name]['metrics']
        
        return results
    
    def create_residual_plots(self, y_true, y_pred, title="Residual Analysis"):
        """Create residual plots for regression models"""
        residuals = y_true - y_pred
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Predicted vs Actual',
                'Residuals vs Predicted', 
                'Residual Distribution',
                'Q-Q Plot'
            ]
        )
        
        # Predicted vs Actual
        fig.add_trace(
            go.Scatter(
                x=y_pred, y=y_true,
                mode='markers',
                name='Predictions',
                opacity=0.6
            ),
            row=1, col=1
        )
        
        # Perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=1
        )
        
        # Residuals vs Predicted
        fig.add_trace(
            go.Scatter(
                x=y_pred, y=residuals,
                mode='markers',
                name='Residuals',
                opacity=0.6
            ),
            row=1, col=2
        )
        
        # Zero line
        fig.add_trace(
            go.Scatter(
                x=[min(y_pred), max(y_pred)], y=[0, 0],
                mode='lines',
                name='Zero Line',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=2
        )
        
        # Residual Distribution
        fig.add_trace(
            go.Histogram(
                x=residuals,
                name='Residual Distribution',
                nbinsx=30
            ),
            row=2, col=1
        )
        
        # Q-Q Plot (approximate)
        from scipy import stats
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
        sample_quantiles = np.sort(residuals)
        
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles, y=sample_quantiles,
                mode='markers',
                name='Q-Q Plot',
                opacity=0.6
            ),
            row=2, col=2
        )
        
        # Q-Q line
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles, y=theoretical_quantiles,
                mode='lines',
                name='Q-Q Line',
                line=dict(color='red', dash='dash')
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=title,
            height=800,
            showlegend=True
        )
        
        return fig
    
    def create_model_comparison_plot(self, results):
        """Create model comparison visualization"""
        models = list(results.keys())
        metrics = ['test_r2', 'test_rmse', 'test_mae']
        
        fig = make_subplots(
            rows=1, cols=len(metrics),
            subplot_titles=metrics
        )
        
        for i, metric in enumerate(metrics):
            values = [results[model]['metrics'][metric] for model in models]
            
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=values,
                    name=metric,
                    showlegend=False
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title="Model Performance Comparison",
            height=400
        )
        
        return fig
    
    def create_feature_importance_plot(self, model, feature_names):
        """Create feature importance plot for tree-based models"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            
            # Sort features by importance
            indices = np.argsort(importance)[::-1]
            sorted_features = [feature_names[i] for i in indices]
            sorted_importance = importance[indices]
            
            fig = go.Figure([
                go.Bar(
                    x=sorted_importance[:20],  # Top 20 features
                    y=sorted_features[:20],
                    orientation='h',
                    marker=dict(
                        color=sorted_importance[:20],
                        colorscale='viridis'
                    )
                )
            ])
            
            fig.update_layout(
                title='Feature Importance (Top 20)',
                xaxis_title='Importance Score',
                yaxis_title='Features',
                height=600
            )
            
            return fig
        else:
            return None
    
    def create_learning_curve(self, model, X, y, cv=5):
        """Create learning curve visualization"""
        from sklearn.model_selection import learning_curve
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='r2'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        fig = go.Figure()
        
        # Training scores
        fig.add_trace(
            go.Scatter(
                x=train_sizes,
                y=train_mean,
                mode='lines+markers',
                name='Training Score',
                line=dict(color='blue'),
                error_y=dict(type='data', array=train_std, visible=True)
            )
        )
        
        # Validation scores
        fig.add_trace(
            go.Scatter(
                x=train_sizes,
                y=val_mean,
                mode='lines+markers',
                name='Validation Score',
                line=dict(color='red'),
                error_y=dict(type='data', array=val_std, visible=True)
            )
        )
        
        fig.update_layout(
            title='Learning Curve',
            xaxis_title='Training Set Size',
            yaxis_title='R² Score',
            height=500
        )
        
        return fig
    
    def create_cross_validation_plot(self, model, X, y, cv=5):
        """Create cross-validation results visualization"""
        scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Box(
                y=scores,
                name='CV Scores',
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            )
        )
        
        fig.update_layout(
            title=f'Cross-Validation Scores (CV={cv})',
            yaxis_title='R² Score',
            height=400
        )
        
        return fig
    
    def create_prediction_interval_plot(self, model, X, y, confidence=0.95):
        """Create prediction interval visualization"""
        if hasattr(model, 'predict'):
            predictions = model.predict(X)
            
            # Calculate prediction intervals (simplified approach)
            residuals = y - predictions
            std_residual = np.std(residuals)
            z_score = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
            
            upper_bound = predictions + z_score * std_residual
            lower_bound = predictions - z_score * std_residual
            
            # Sort for better visualization
            sorted_indices = np.argsort(predictions)
            sorted_pred = predictions[sorted_indices]
            sorted_actual = y.iloc[sorted_indices] if hasattr(y, 'iloc') else y[sorted_indices]
            sorted_upper = upper_bound[sorted_indices]
            sorted_lower = lower_bound[sorted_indices]
            
            fig = go.Figure()
            
            # Prediction intervals
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(sorted_pred))),
                    y=sorted_upper,
                    mode='lines',
                    name=f'Upper Bound ({confidence*100}%)',
                    line=dict(color='red', dash='dash'),
                    fill=None
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(sorted_pred))),
                    y=sorted_lower,
                    mode='lines',
                    name=f'Lower Bound ({confidence*100}%)',
                    line=dict(color='red', dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.2)'
                )
            )
            
            # Predictions
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(sorted_pred))),
                    y=sorted_pred,
                    mode='lines',
                    name='Predictions',
                    line=dict(color='blue')
                )
            )
            
            # Actual values
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(sorted_actual))),
                    y=sorted_actual,
                    mode='markers',
                    name='Actual Values',
                    marker=dict(color='green', size=4, opacity=0.6)
                )
            )
            
            fig.update_layout(
                title=f'Prediction Intervals ({confidence*100}% Confidence)',
                xaxis_title='Sample Index (sorted by prediction)',
                yaxis_title='Value',
                height=500
            )
            
            return fig
        
        return None