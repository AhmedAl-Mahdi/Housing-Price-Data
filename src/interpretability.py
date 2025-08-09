"""
Model Interpretability Module
Contains functions for SHAP, LIME, and other interpretability tools
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class ModelInterpreter:
    def __init__(self, model, X_train, X_test=None):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test if X_test is not None else X_train
        self.feature_names = list(X_train.columns)
        
    def get_shap_values(self, sample_size=100):
        """Calculate SHAP values for model interpretability"""
        try:
            import shap
            
            # Sample data for faster computation
            if len(self.X_test) > sample_size:
                sample_indices = np.random.choice(len(self.X_test), sample_size, replace=False)
                X_sample = self.X_test.iloc[sample_indices]
            else:
                X_sample = self.X_test
            
            # Create explainer based on model type
            model_type = str(type(self.model).__name__)
            
            if 'RandomForest' in model_type or 'GradientBoosting' in model_type or hasattr(self.model, 'tree_'):
                # Tree-based models
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X_sample)
            elif 'Linear' in model_type or hasattr(self.model, 'coef_'):
                # Linear models
                explainer = shap.LinearExplainer(self.model, self.X_train)
                shap_values = explainer.shap_values(X_sample)
            else:
                # Fallback to KernelExplainer for other models
                explainer = shap.KernelExplainer(self.model.predict, self.X_train.sample(min(100, len(self.X_train))))
                shap_values = explainer.shap_values(X_sample)
            
            return shap_values, X_sample, explainer
            
        except ImportError:
            return None, None, None
        except Exception as e:
            print(f"Error calculating SHAP values: {str(e)}")
            return None, None, None
    
    def create_shap_summary_plot(self, shap_values=None, X_sample=None):
        """Create SHAP summary plot"""
        if shap_values is None or X_sample is None:
            shap_values, X_sample, _ = self.get_shap_values()
        
        if shap_values is None:
            return self.create_feature_importance_fallback()
        
        # Calculate mean absolute SHAP values for feature importance
        mean_shap = np.mean(np.abs(shap_values), axis=0)
        
        # Create feature importance plot
        feature_importance = dict(zip(self.feature_names, mean_shap))
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        features, importance = zip(*sorted_features[:20])  # Top 20 features
        
        fig = go.Figure([
            go.Bar(
                x=list(importance),
                y=list(features),
                orientation='h',
                marker=dict(
                    color=importance,
                    colorscale='viridis'
                )
            )
        ])
        
        fig.update_layout(
            title='SHAP Feature Importance',
            xaxis_title='Mean |SHAP Value|',
            yaxis_title='Features',
            height=max(400, len(features) * 25)
        )
        
        return fig
    
    def create_shap_waterfall_plot(self, instance_idx=0, shap_values=None, X_sample=None):
        """Create SHAP waterfall plot for a single prediction"""
        if shap_values is None or X_sample is None:
            shap_values, X_sample, _ = self.get_shap_values()
        
        if shap_values is None:
            return None
        
        if instance_idx >= len(shap_values):
            instance_idx = 0
        
        # Get SHAP values for specific instance
        instance_shap = shap_values[instance_idx]
        instance_features = X_sample.iloc[instance_idx]
        
        # Base value (expected value)
        base_value = np.mean(self.model.predict(self.X_train))
        
        # Sort features by absolute SHAP value
        feature_contribution = list(zip(self.feature_names, instance_shap, instance_features))
        feature_contribution.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Take top features
        top_features = feature_contribution[:15]
        
        features = [f"{name}={value:.2f}" for name, _, value in top_features]
        shap_vals = [contrib for _, contrib, _ in top_features]
        
        # Calculate cumulative values for waterfall
        cumulative = [base_value]
        for val in shap_vals:
            cumulative.append(cumulative[-1] + val)
        
        # Prediction value
        prediction = cumulative[-1]
        
        fig = go.Figure()
        
        # Base value
        fig.add_trace(
            go.Bar(
                x=['Base Value'],
                y=[base_value],
                name='Base Value',
                marker_color='blue'
            )
        )
        
        # Feature contributions
        colors = ['green' if val > 0 else 'red' for val in shap_vals]
        fig.add_trace(
            go.Bar(
                x=features,
                y=shap_vals,
                name='SHAP Values',
                marker_color=colors
            )
        )
        
        # Final prediction
        fig.add_trace(
            go.Bar(
                x=['Prediction'],
                y=[prediction],
                name='Final Prediction',
                marker_color='darkblue'
            )
        )
        
        fig.update_layout(
            title=f'SHAP Waterfall Plot - Instance {instance_idx}',
            xaxis_title='Features',
            yaxis_title='Value',
            height=600,
            xaxis={'tickangle': 45}
        )
        
        return fig
    
    def create_shap_dependence_plot(self, feature_name, shap_values=None, X_sample=None):
        """Create SHAP dependence plot for a specific feature"""
        if shap_values is None or X_sample is None:
            shap_values, X_sample, _ = self.get_shap_values()
        
        if shap_values is None or feature_name not in self.feature_names:
            return None
        
        feature_idx = self.feature_names.index(feature_name)
        feature_values = X_sample[feature_name]
        feature_shap = shap_values[:, feature_idx]
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=feature_values,
                y=feature_shap,
                mode='markers',
                marker=dict(
                    color=feature_shap,
                    colorscale='viridis',
                    showscale=True,
                    colorbar=dict(title="SHAP Value")
                ),
                name='SHAP Values'
            )
        )
        
        fig.update_layout(
            title=f'SHAP Dependence Plot - {feature_name}',
            xaxis_title=f'{feature_name} Value',
            yaxis_title='SHAP Value',
            height=500
        )
        
        return fig
    
    def get_lime_explanation(self, instance_idx=0, sample_size=100):
        """Get LIME explanation for a single instance"""
        try:
            from lime.lime_tabular import LimeTabularExplainer
            
            # Sample data for faster computation
            if len(self.X_test) > sample_size:
                sample_indices = np.random.choice(len(self.X_test), sample_size, replace=False)
                X_sample = self.X_test.iloc[sample_indices]
            else:
                X_sample = self.X_test
            
            if instance_idx >= len(X_sample):
                instance_idx = 0
            
            # Create LIME explainer
            explainer = LimeTabularExplainer(
                self.X_train.values,
                feature_names=self.feature_names,
                mode='regression',
                discretize_continuous=True
            )
            
            # Explain instance
            explanation = explainer.explain_instance(
                X_sample.iloc[instance_idx].values,
                self.model.predict,
                num_features=len(self.feature_names)
            )
            
            return explanation, X_sample
            
        except ImportError:
            return None, None
        except Exception as e:
            print(f"Error creating LIME explanation: {str(e)}")
            return None, None
    
    def create_lime_plot(self, instance_idx=0):
        """Create LIME explanation plot"""
        explanation, X_sample = self.get_lime_explanation(instance_idx)
        
        if explanation is None:
            return self.create_feature_importance_fallback()
        
        # Extract feature importance from LIME explanation
        lime_data = explanation.as_list()
        features, importance = zip(*lime_data)
        
        # Sort by absolute importance
        sorted_data = sorted(zip(features, importance), key=lambda x: abs(x[1]), reverse=True)
        features, importance = zip(*sorted_data)
        
        colors = ['green' if val > 0 else 'red' for val in importance]
        
        fig = go.Figure([
            go.Bar(
                x=list(importance),
                y=list(features),
                orientation='h',
                marker_color=colors
            )
        ])
        
        fig.update_layout(
            title=f'LIME Explanation - Instance {instance_idx}',
            xaxis_title='Feature Importance',
            yaxis_title='Features',
            height=max(400, len(features) * 25)
        )
        
        return fig
    
    def create_partial_dependence_plot(self, feature_name):
        """Create partial dependence plot for a feature"""
        try:
            from sklearn.inspection import partial_dependence
            
            if feature_name not in self.feature_names:
                return None
            
            feature_idx = self.feature_names.index(feature_name)
            
            # Calculate partial dependence
            pdp_result = partial_dependence(
                self.model, 
                self.X_train, 
                features=[feature_idx],
                kind='average'
            )
            
            # Extract values and grid
            pdp_values = pdp_result['average'][0]
            feature_grid = pdp_result['grid'][0]
            
            fig = go.Figure()
            
            fig.add_trace(
                go.Scatter(
                    x=feature_grid,
                    y=pdp_values,
                    mode='lines+markers',
                    name='Partial Dependence',
                    line=dict(color='blue', width=3)
                )
            )
            
            fig.update_layout(
                title=f'Partial Dependence Plot - {feature_name}',
                xaxis_title=f'{feature_name}',
                yaxis_title='Partial Dependence',
                height=500
            )
            
            return fig
            
        except ImportError:
            return None
        except Exception as e:
            print(f"Error creating partial dependence plot: {str(e)}")
            return None
    
    def create_feature_interaction_plot(self, feature1, feature2):
        """Create feature interaction plot"""
        try:
            from sklearn.inspection import partial_dependence
            import sklearn
            
            if feature1 not in self.feature_names or feature2 not in self.feature_names:
                return None
            
            feature1_idx = self.feature_names.index(feature1)
            feature2_idx = self.feature_names.index(feature2)
            
            # Calculate 2D partial dependence
            try:
                pdp_result = partial_dependence(
                    self.model,
                    self.X_train,
                    features=[feature1_idx, feature2_idx],
                    kind='average'
                )
                
                # Handle different result structures based on sklearn version
                if hasattr(pdp_result, 'average'):
                    # Newer sklearn versions return a Bunch object
                    pdp_values = pdp_result.average[0]
                    feature1_grid = pdp_result.grid_values[0]
                    feature2_grid = pdp_result.grid_values[1]
                elif isinstance(pdp_result, dict):
                    # Dictionary format
                    pdp_values = pdp_result['average'][0]
                    feature1_grid = pdp_result['grid'][0]
                    feature2_grid = pdp_result['grid'][1]
                elif isinstance(pdp_result, tuple):
                    # Tuple format
                    pdp_values, axes = pdp_result
                    if len(pdp_values.shape) > 2:
                        pdp_values = pdp_values[0]
                    feature1_grid = axes[0]
                    feature2_grid = axes[1]
                else:
                    print(f"Unexpected result format: {type(pdp_result)}")
                    return None
                    
            except Exception as api_error:
                print(f"API error: {str(api_error)}")
                return None
            
            # Create contour plot
            fig = go.Figure(data=go.Contour(
                x=feature1_grid,
                y=feature2_grid,
                z=pdp_values,
                colorscale='viridis',
                showscale=True,
                colorbar=dict(title="Partial Dependence"),
                contours=dict(
                    showlabels=True,
                    labelfont=dict(size=12, color='white')
                )
            ))
            
            fig.update_layout(
                title=f'Feature Interaction: {feature1} vs {feature2}',
                xaxis_title=feature1,
                yaxis_title=feature2,
                height=500,
                width=600
            )
            
            return fig
            
        except ImportError:
            print("sklearn.inspection.partial_dependence not available")
            return None
        except Exception as e:
            print(f"Error creating feature interaction plot: {str(e)}")
            return None
    
    def create_feature_importance_fallback(self):
        """Fallback feature importance plot when SHAP/LIME not available"""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            feature_importance = dict(zip(self.feature_names, importance))
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_.flatten())
            feature_importance = dict(zip(self.feature_names, importance))
        else:
            return None
        
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        features, importance = zip(*sorted_features[:20])
        
        fig = go.Figure([
            go.Bar(
                x=list(importance),
                y=list(features),
                orientation='h',
                marker=dict(
                    color=importance,
                    colorscale='viridis'
                )
            )
        ])
        
        fig.update_layout(
            title='Feature Importance (Model-based)',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            height=max(400, len(features) * 25)
        )
        
        return fig