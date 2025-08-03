"""
EDA Module
Contains functions for exploratory data analysis and visualization
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class EDAVisualizer:
    def __init__(self, data):
        self.data = data
        
    def create_histogram(self, column, bins=30, title=None):
        """Create interactive histogram"""
        if title is None:
            title = f'Distribution of {column}'
            
        fig = px.histogram(
            self.data, 
            x=column, 
            nbins=bins,
            title=title,
            marginal="box"
        )
        
        fig.update_layout(
            xaxis_title=column,
            yaxis_title='Frequency',
            showlegend=False
        )
        
        return fig
    
    def create_scatter_plot(self, x_col, y_col, color_col=None, title=None):
        """Create interactive scatter plot"""
        if title is None:
            title = f'{y_col} vs {x_col}'
            
        fig = px.scatter(
            self.data,
            x=x_col,
            y=y_col,
            color=color_col,
            title=title,
            trendline="ols" if color_col is None else None
        )
        
        return fig
    
    def create_box_plot(self, column, group_by=None, title=None):
        """Create interactive box plot"""
        if title is None:
            title = f'Box Plot of {column}'
            if group_by:
                title += f' by {group_by}'
                
        if group_by:
            fig = px.box(
                self.data,
                x=group_by,
                y=column,
                title=title
            )
        else:
            fig = px.box(
                self.data,
                y=column,
                title=title
            )
            
        return fig
    
    def create_correlation_heatmap(self, title="Correlation Matrix"):
        """Create correlation heatmap"""
        numeric_data = self.data.select_dtypes(include=[np.number])
        corr_matrix = numeric_data.corr()
        
        fig = px.imshow(
            corr_matrix,
            title=title,
            color_continuous_scale='RdBu_r',
            aspect="auto",
            text_auto=True
        )
        
        fig.update_layout(
            width=800,
            height=600
        )
        
        return fig
    
    def create_pair_plot(self, columns=None, max_cols=5):
        """Create pair plot for selected columns"""
        if columns is None:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            columns = numeric_cols[:max_cols] if len(numeric_cols) > max_cols else numeric_cols
        
        n_cols = len(columns)
        fig = make_subplots(
            rows=n_cols, 
            cols=n_cols,
            subplot_titles=[f"{col1} vs {col2}" for col1 in columns for col2 in columns]
        )
        
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i == j:
                    # Diagonal: histogram
                    fig.add_trace(
                        go.Histogram(x=self.data[col1], name=f"Hist {col1}"),
                        row=i+1, col=j+1
                    )
                else:
                    # Off-diagonal: scatter plot
                    fig.add_trace(
                        go.Scatter(
                            x=self.data[col2], 
                            y=self.data[col1],
                            mode='markers',
                            name=f"{col1} vs {col2}",
                            showlegend=False
                        ),
                        row=i+1, col=j+1
                    )
        
        fig.update_layout(
            title="Pair Plot",
            height=200 * n_cols,
            showlegend=False
        )
        
        return fig
    
    def create_missing_data_plot(self):
        """Create missing data visualization"""
        missing_data = self.data.isnull().sum()
        missing_percentage = (missing_data / len(self.data)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing_Count': missing_data.values,
            'Missing_Percentage': missing_percentage.values
        }).sort_values('Missing_Count', ascending=False)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Missing Values Count', 'Missing Values Percentage'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Count plot
        fig.add_trace(
            go.Bar(
                x=missing_df['Column'],
                y=missing_df['Missing_Count'],
                name='Count'
            ),
            row=1, col=1
        )
        
        # Percentage plot
        fig.add_trace(
            go.Bar(
                x=missing_df['Column'],
                y=missing_df['Missing_Percentage'],
                name='Percentage'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Missing Data Analysis",
            showlegend=False,
            height=500
        )
        
        return fig
    
    def create_outlier_plot(self, column, method='iqr'):
        """Create outlier detection visualization"""
        data = self.data[column].dropna()
        
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data < lower_bound) | (data > upper_bound)]
        else:
            z_scores = np.abs((data - data.mean()) / data.std())
            outliers = data[z_scores > 3]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[f'Box Plot - {column}', f'Histogram with Outliers - {column}']
        )
        
        # Box plot
        fig.add_trace(
            go.Box(y=data, name=column),
            row=1, col=1
        )
        
        # Histogram with outliers highlighted
        fig.add_trace(
            go.Histogram(x=data, name='Normal Data', opacity=0.7),
            row=1, col=2
        )
        
        if len(outliers) > 0:
            fig.add_trace(
                go.Histogram(x=outliers, name='Outliers', opacity=0.7),
                row=1, col=2
            )
        
        fig.update_layout(
            title=f"Outlier Analysis for {column}",
            height=400,
            showlegend=True
        )
        
        return fig
    
    def create_distribution_comparison(self, column, group_col):
        """Create distribution comparison across groups"""
        fig = px.violin(
            self.data,
            x=group_col,
            y=column,
            box=True,
            title=f'Distribution of {column} by {group_col}'
        )
        
        return fig
    
    def create_feature_importance_plot(self, feature_importance_dict):
        """Create feature importance visualization"""
        features = list(feature_importance_dict.keys())
        importance = list(feature_importance_dict.values())
        
        # Sort by importance
        sorted_data = sorted(zip(features, importance), key=lambda x: x[1], reverse=True)
        features, importance = zip(*sorted_data)
        
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
            title='Feature Importance',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            height=max(400, len(features) * 25)
        )
        
        return fig
    
    def create_summary_statistics_table(self):
        """Create summary statistics table"""
        numeric_data = self.data.select_dtypes(include=[np.number])
        stats = numeric_data.describe().round(2)
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Statistic'] + list(stats.columns),
                fill_color='paleturquoise',
                align='left'
            ),
            cells=dict(
                values=[stats.index] + [stats[col] for col in stats.columns],
                fill_color='lavender',
                align='left'
            )
        )])
        
        fig.update_layout(
            title="Summary Statistics",
            height=400
        )
        
        return fig