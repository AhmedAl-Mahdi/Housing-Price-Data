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
import base64
import io
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
            if hasattr(self.model, 'tree_'):
                # Tree-based models
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X_sample)
            else:
                # Linear models or others
                explainer = shap.LinearExplainer(self.model, self.X_train)
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
            
            if feature1 not in self.feature_names or feature2 not in self.feature_names:
                return None
            
            feature1_idx = self.feature_names.index(feature1)
            feature2_idx = self.feature_names.index(feature2)
            
            # Calculate 2D partial dependence
            pdp_result = partial_dependence(
                self.model,
                self.X_train,
                features=[feature1_idx, feature2_idx],
                kind='average'
            )
            
            # Extract values and grids
            pdp_values = pdp_result['average'][0]
            feature1_grid = pdp_result['grid'][0]
            feature2_grid = pdp_result['grid'][1]
            
            fig = go.Figure(data=go.Contour(
                x=feature1_grid,
                y=feature2_grid,
                z=pdp_values,
                colorscale='viridis',
                showscale=True
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
    
    def create_interactive_shap_summary(self, shap_values=None, X_sample=None):
        """Create enhanced interactive SHAP summary plot with multiple views"""
        if shap_values is None or X_sample is None:
            shap_values, X_sample, _ = self.get_shap_values()
        
        if shap_values is None:
            return self.create_feature_importance_fallback()
        
        # Calculate feature importance
        mean_shap = np.mean(np.abs(shap_values), axis=0)
        feature_importance = dict(zip(self.feature_names, mean_shap))
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Create subplots for multiple views
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Feature Importance', 'Value Distribution', 'Impact Analysis', 'Feature Interactions'),
            specs=[[{"type": "bar"}, {"type": "violin"}],
                   [{"type": "scatter"}, {"type": "heatmap"}]]
        )
        
        # 1. Feature Importance (top 10)
        top_features = sorted_features[:10]
        features, importance = zip(*top_features)
        
        fig.add_trace(
            go.Bar(
                x=list(importance),
                y=list(features),
                orientation='h',
                name='Feature Importance',
                marker=dict(color=importance, colorscale='viridis'),
                hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. SHAP value distribution for top feature
        top_feature = features[0]
        top_feature_idx = self.feature_names.index(top_feature)
        shap_vals_top = shap_values[:, top_feature_idx]
        
        fig.add_trace(
            go.Violin(
                y=shap_vals_top,
                name=f'{top_feature} SHAP values',
                box_visible=True,
                meanline_visible=True,
                fillcolor='lightblue',
                opacity=0.6,
                hovertemplate='SHAP Value: %{y:.4f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # 3. Feature value vs SHAP value scatter for top feature
        feature_vals_top = X_sample[top_feature]
        
        fig.add_trace(
            go.Scatter(
                x=feature_vals_top,
                y=shap_vals_top,
                mode='markers',
                name=f'{top_feature} Impact',
                marker=dict(
                    size=8,
                    color=shap_vals_top,
                    colorscale='RdYlBu',
                    showscale=True,
                    opacity=0.7
                ),
                hovertemplate='<b>%{text}</b><br>Value: %{x}<br>SHAP: %{y:.4f}<extra></extra>',
                text=[f'Property {i}' for i in range(len(feature_vals_top))]
            ),
            row=2, col=1
        )
        
        # 4. Feature correlation heatmap (top 5 features)
        top_5_features = [f[0] for f in sorted_features[:5]]
        correlation_matrix = X_sample[top_5_features].corr()
        
        fig.add_trace(
            go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                showscale=True,
                hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="Interactive SHAP Analysis Dashboard",
            showlegend=True,
            title_x=0.5
        )
        
        return fig
    
    def create_shap_decision_plot(self, shap_values=None, X_sample=None, max_instances=10):
        """Create SHAP decision plot showing prediction paths"""
        if shap_values is None or X_sample is None:
            shap_values, X_sample, _ = self.get_shap_values()
        
        if shap_values is None:
            return None
        
        # Calculate cumulative SHAP values for decision paths
        base_value = np.mean(self.model.predict(self.X_train))
        
        # Select subset of instances
        n_instances = min(max_instances, len(shap_values))
        indices = np.random.choice(len(shap_values), n_instances, replace=False)
        
        # Get top features by average importance
        mean_shap = np.mean(np.abs(shap_values), axis=0)
        top_features_idx = np.argsort(mean_shap)[-10:][::-1]  # Top 10 features
        
        fig = go.Figure()
        
        # Create decision paths for each instance
        for i, idx in enumerate(indices):
            instance_shap = shap_values[idx, top_features_idx]
            cumulative = np.cumsum(np.concatenate([[base_value], instance_shap]))
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(cumulative))),
                    y=cumulative,
                    mode='lines+markers',
                    name=f'Property {idx}',
                    line=dict(width=2),
                    marker=dict(size=6),
                    hovertemplate=f'Property {idx}<br>Step: %{{x}}<br>Cumulative: %{{y:.0f}}<extra></extra>'
                )
            )
        
        # Add feature names to x-axis
        feature_names_subset = ['Base'] + [self.feature_names[i] for i in top_features_idx]
        
        fig.update_layout(
            title='SHAP Decision Plot - Prediction Paths',
            xaxis_title='Decision Steps',
            yaxis_title='Cumulative Prediction Value',
            height=600,
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(len(feature_names_subset))),
                ticktext=feature_names_subset,
                tickangle=45
            ),
            hovermode='closest'
        )
        
        return fig
    
    def create_shap_beeswarm_plot(self, shap_values=None, X_sample=None):
        """Create SHAP beeswarm plot (alternative to summary plot)"""
        if shap_values is None or X_sample is None:
            shap_values, X_sample, _ = self.get_shap_values()
        
        if shap_values is None:
            return None
        
        # Calculate feature importance and select top features
        mean_shap = np.mean(np.abs(shap_values), axis=0)
        top_features_idx = np.argsort(mean_shap)[-15:][::-1]  # Top 15 features
        
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for i, feature_idx in enumerate(top_features_idx):
            feature_name = self.feature_names[feature_idx]
            shap_vals = shap_values[:, feature_idx]
            feature_vals = X_sample.iloc[:, feature_idx]
            
            # Add jitter to y-axis for beeswarm effect
            y_jitter = np.random.normal(0, 0.1, len(shap_vals))
            y_positions = np.full(len(shap_vals), len(top_features_idx) - i - 1) + y_jitter
            
            fig.add_trace(
                go.Scatter(
                    x=shap_vals,
                    y=y_positions,
                    mode='markers',
                    name=feature_name,
                    marker=dict(
                        size=6,
                        color=feature_vals,
                        colorscale='RdYlBu',
                        opacity=0.7,
                        line=dict(width=0.5, color='white')
                    ),
                    hovertemplate=f'<b>{feature_name}</b><br>SHAP: %{{x:.4f}}<br>Value: %{{marker.color}}<extra></extra>',
                    showlegend=False
                )
            )
        
        # Update layout
        fig.update_layout(
            title='SHAP Beeswarm Plot - Feature Impact Distribution',
            xaxis_title='SHAP Value (impact on model output)',
            yaxis_title='Features',
            height=600,
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(len(top_features_idx))),
                ticktext=[self.feature_names[i] for i in top_features_idx][::-1]
            ),
            xaxis=dict(zeroline=True, zerolinecolor='black', zerolinewidth=2),
            hovermode='closest'
        )
        
        return fig
    
    def create_shap_force_plot_html(self, instance_idx=0, shap_values=None, X_sample=None):
        """Create native SHAP force plot as HTML for embedding"""
        try:
            import shap
            
            if shap_values is None or X_sample is None:
                shap_values, X_sample, explainer = self.get_shap_values()
            else:
                explainer = None
                
            if shap_values is None:
                return None
            
            if instance_idx >= len(shap_values):
                instance_idx = 0
            
            # Get expected value (base value)
            expected_value = np.mean(self.model.predict(self.X_train))
            
            # Create force plot using SHAP
            force_plot = shap.force_plot(
                expected_value,
                shap_values[instance_idx],
                X_sample.iloc[instance_idx],
                matplotlib=False,
                show=False
            )
            
            # Convert to HTML
            shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
            return shap_html
            
        except Exception as e:
            print(f"Error creating SHAP force plot HTML: {str(e)}")
            return None
    
    def create_shap_clustering_plot(self, shap_values=None, X_sample=None):
        """Create SHAP clustering plot to identify similar prediction patterns"""
        if shap_values is None or X_sample is None:
            shap_values, X_sample, _ = self.get_shap_values()
        
        if shap_values is None:
            return None
        
        try:
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
            
            # Perform clustering on SHAP values
            n_clusters = min(5, len(shap_values) // 5)  # Adaptive number of clusters
            if n_clusters < 2:
                n_clusters = 2
                
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(shap_values)
            
            # Reduce dimensionality for visualization
            pca = PCA(n_components=2)
            shap_2d = pca.fit_transform(shap_values)
            
            # Create scatter plot
            fig = go.Figure()
            
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            
            for cluster in range(n_clusters):
                mask = cluster_labels == cluster
                fig.add_trace(
                    go.Scatter(
                        x=shap_2d[mask, 0],
                        y=shap_2d[mask, 1],
                        mode='markers',
                        name=f'Cluster {cluster + 1}',
                        marker=dict(
                            size=8,
                            color=colors[cluster % len(colors)],
                            opacity=0.7
                        ),
                        hovertemplate=f'Cluster {cluster + 1}<br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>'
                    )
                )
            
            # Add cluster centers
            centers_2d = pca.transform(kmeans.cluster_centers_)
            fig.add_trace(
                go.Scatter(
                    x=centers_2d[:, 0],
                    y=centers_2d[:, 1],
                    mode='markers',
                    name='Cluster Centers',
                    marker=dict(
                        size=15,
                        color='black',
                        symbol='x',
                        line=dict(width=2, color='white')
                    ),
                    hovertemplate='Cluster Center<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
                )
            )
            
            fig.update_layout(
                title='SHAP Values Clustering - Similar Prediction Patterns',
                xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)',
                height=600,
                hovermode='closest'
            )
            
            return fig, cluster_labels
            
        except ImportError:
            return None, None
        except Exception as e:
            print(f"Error creating clustering plot: {str(e)}")
            return None, None
        
        return fig