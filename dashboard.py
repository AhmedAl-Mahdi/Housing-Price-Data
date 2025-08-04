"""
Interactive Machine Learning Dashboard for Housing Price Analysis
A comprehensive dashboard for data exploration, model evaluation, and interpretability
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processing import DataProcessor
from eda_module import EDAVisualizer
from model_evaluation import ModelEvaluator
from interpretability import ModelInterpreter

# Configure page
st.set_page_config(
    page_title="Housing Price ML Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e8b57;
        border-bottom: 2px solid #2e8b57;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the housing dataset"""
    try:
        data = pd.read_csv('Housing_Price_Data.csv')
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def initialize_processor(data):
    """Initialize and cache the data processor"""
    processor = DataProcessor()
    processor.original_data = data
    processor.processed_data = data.copy()
    return processor

def main():
    # Header
    st.markdown('<h1 class="main-header">🏠 Housing Price ML Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**A comprehensive tool for exploratory analysis, model evaluation, and interpretability**")
    
    # Load data
    data = load_data()
    if data is None:
        st.stop()
    
    # Initialize processor
    processor = initialize_processor(data)
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choose a module:",
        [
            "📊 Data Overview",
            "🔍 Exploratory Data Analysis", 
            "🔧 Data Preprocessing",
            "🤖 Model Training & Evaluation",
            "🔬 Model Interpretability",
            "🏠 Price Prediction",
            "📈 Interactive Features"
        ]
    )
    
    # Data Overview
    if page == "📊 Data Overview":
        show_data_overview(data, processor)
    
    # EDA Module
    elif page == "🔍 Exploratory Data Analysis":
        show_eda_module(data, processor)
    
    # Preprocessing Module
    elif page == "🔧 Data Preprocessing":
        show_preprocessing_module(data, processor)
    
    # Model Evaluation
    elif page == "🤖 Model Training & Evaluation":
        show_model_evaluation(data, processor)
    
    # Model Interpretability
    elif page == "🔬 Model Interpretability":
        show_interpretability_module(data, processor)
    
    # Price Prediction
    elif page == "🏠 Price Prediction":
        show_price_prediction(data, processor)
    
    # Interactive Features
    elif page == "📈 Interactive Features":
        show_interactive_features(data, processor)

def show_data_overview(data, processor):
    """Display data overview and summary statistics"""
    st.markdown('<h2 class="section-header">Data Overview</h2>', unsafe_allow_html=True)
    
    # Data info metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(data):,}")
    with col2:
        st.metric("Features", len(data.columns))
    with col3:
        st.metric("Missing Values", data.isnull().sum().sum())
    with col4:
        st.metric("Avg Price", f"${data['price'].mean():,.0f}")
    
    # Dataset preview
    st.subheader("Dataset Preview")
    st.dataframe(data.head(10), use_container_width=True)
    
    # Data types and info
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Types")
        data_types = pd.DataFrame({
            'Column': data.columns,
            'Type': data.dtypes,
            'Non-Null Count': data.count(),
            'Missing': data.isnull().sum()
        })
        st.dataframe(data_types, use_container_width=True)
    
    with col2:
        st.subheader("Summary Statistics")
        st.dataframe(data.describe(), use_container_width=True)
    
    # Missing data visualization
    if data.isnull().sum().sum() > 0:
        st.subheader("Missing Data Analysis")
        visualizer = EDAVisualizer(data)
        missing_fig = visualizer.create_missing_data_plot()
        st.plotly_chart(missing_fig, use_container_width=True)

def show_eda_module(data, processor):
    """Display exploratory data analysis tools"""
    st.markdown('<h2 class="section-header">Exploratory Data Analysis</h2>', unsafe_allow_html=True)
    
    visualizer = EDAVisualizer(data)
    
    # Sidebar controls
    st.sidebar.subheader("📊 EDA Controls")
    
    # Analysis type selection
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type:",
        [
            "Target Variable Analysis",
            "Feature Distributions", 
            "Correlation Analysis",
            "Outlier Detection",
            "Categorical Analysis"
        ]
    )
    
    if analysis_type == "Target Variable Analysis":
        st.subheader("Housing Price Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Price histogram
            hist_fig = visualizer.create_histogram('price', title="Price Distribution")
            st.plotly_chart(hist_fig, use_container_width=True)
        
        with col2:
            # Price box plot
            box_fig = visualizer.create_box_plot('price', title="Price Box Plot")
            st.plotly_chart(box_fig, use_container_width=True)
        
        # Price statistics
        st.subheader("Price Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean", f"${data['price'].mean():,.0f}")
        with col2:
            st.metric("Median", f"${data['price'].median():,.0f}")
        with col3:
            st.metric("Std Dev", f"${data['price'].std():,.0f}")
        with col4:
            st.metric("Range", f"${data['price'].max() - data['price'].min():,.0f}")
    
    elif analysis_type == "Feature Distributions":
        st.subheader("Feature Distribution Analysis")
        
        # Feature selection
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        selected_feature = st.selectbox("Select Feature:", numeric_columns)
        
        if selected_feature:
            col1, col2 = st.columns(2)
            
            with col1:
                hist_fig = visualizer.create_histogram(selected_feature)
                st.plotly_chart(hist_fig, use_container_width=True)
            
            with col2:
                box_fig = visualizer.create_box_plot(selected_feature)
                st.plotly_chart(box_fig, use_container_width=True)
            
            # Feature vs Price scatter plot
            if selected_feature != 'price':
                scatter_fig = visualizer.create_scatter_plot(selected_feature, 'price')
                st.plotly_chart(scatter_fig, use_container_width=True)
    
    elif analysis_type == "Correlation Analysis":
        st.subheader("Correlation Analysis")
        
        # Correlation heatmap
        corr_fig = visualizer.create_correlation_heatmap()
        st.plotly_chart(corr_fig, use_container_width=True)
        
        # Top correlations with price
        numeric_data = data.select_dtypes(include=[np.number])
        correlations = numeric_data.corr()['price'].drop('price').sort_values(ascending=False)
        
        st.subheader("Correlations with Price")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Positive Correlations:**")
            positive_corr = correlations[correlations > 0].head(5)
            for feature, corr in positive_corr.items():
                st.write(f"• {feature}: {corr:.3f}")
        
        with col2:
            st.write("**Negative Correlations:**")
            negative_corr = correlations[correlations < 0].tail(5)
            for feature, corr in negative_corr.items():
                st.write(f"• {feature}: {corr:.3f}")
    
    elif analysis_type == "Outlier Detection":
        st.subheader("Outlier Detection")
        
        # Feature selection for outlier analysis
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        selected_feature = st.selectbox("Select Feature for Outlier Analysis:", numeric_columns)
        
        if selected_feature:
            # Outlier detection method
            method = st.radio("Detection Method:", ["IQR", "Z-Score"])
            
            outlier_info = processor.detect_outliers(selected_feature, method.lower().replace('-', '_'))
            
            if outlier_info:
                # Display outlier statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Outliers Found", outlier_info['count'])
                with col2:
                    st.metric("Percentage", f"{outlier_info['percentage']:.1f}%")
                with col3:
                    st.metric("Total Values", len(data[selected_feature].dropna()))
                
                # Outlier visualization
                outlier_fig = visualizer.create_outlier_plot(selected_feature, method.lower().replace('-', '_'))
                st.plotly_chart(outlier_fig, use_container_width=True)
    
    elif analysis_type == "Categorical Analysis":
        st.subheader("Categorical Variable Analysis")
        
        categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
        if categorical_columns:
            selected_cat = st.selectbox("Select Categorical Feature:", categorical_columns)
            
            if selected_cat:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Value counts
                    value_counts = data[selected_cat].value_counts()
                    fig = px.bar(x=value_counts.index, y=value_counts.values,
                               title=f"Distribution of {selected_cat}")
                    fig.update_xaxis(title=selected_cat)
                    fig.update_yaxis(title="Count")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Price by category
                    avg_price = data.groupby(selected_cat)['price'].mean().sort_values(ascending=False)
                    fig = px.bar(x=avg_price.index, y=avg_price.values,
                               title=f"Average Price by {selected_cat}")
                    fig.update_xaxis(title=selected_cat)
                    fig.update_yaxis(title="Average Price")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Box plot of price by category
                box_fig = visualizer.create_distribution_comparison('price', selected_cat)
                st.plotly_chart(box_fig, use_container_width=True)

def show_preprocessing_module(data, processor):
    """Display data preprocessing tools"""
    st.markdown('<h2 class="section-header">Data Preprocessing</h2>', unsafe_allow_html=True)
    
    st.sidebar.subheader("🔧 Preprocessing Options")
    
    # Preprocessing options
    preprocess_type = st.sidebar.selectbox(
        "Select Preprocessing Task:",
        [
            "Missing Value Handling",
            "Outlier Treatment", 
            "Feature Scaling",
            "Categorical Encoding",
            "Before/After Comparison"
        ]
    )
    
    if preprocess_type == "Missing Value Handling":
        st.subheader("Missing Value Analysis and Treatment")
        
        missing_summary = data.isnull().sum()
        missing_cols = missing_summary[missing_summary > 0].index.tolist()
        
        if len(missing_cols) > 0:
            st.write("**Columns with Missing Values:**")
            for col in missing_cols:
                missing_count = missing_summary[col]
                missing_pct = (missing_count / len(data)) * 100
                st.write(f"• {col}: {missing_count} missing ({missing_pct:.1f}%)")
            
            # Missing value visualization
            visualizer = EDAVisualizer(data)
            missing_fig = visualizer.create_missing_data_plot()
            st.plotly_chart(missing_fig, use_container_width=True)
            
            # Treatment options
            st.subheader("Treatment Options")
            strategy = st.selectbox("Select Strategy:", ["mean", "median", "mode"])
            
            if st.button("Apply Missing Value Treatment"):
                processor.handle_missing_values(strategy=strategy)
                st.success(f"Missing values handled using {strategy} strategy!")
                
                # Show before/after
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Before:**")
                    st.write(data.isnull().sum())
                with col2:
                    st.write("**After:**")
                    st.write(processor.processed_data.isnull().sum())
        else:
            st.success("✅ No missing values found in the dataset!")
    
    elif preprocess_type == "Feature Scaling":
        st.subheader("Feature Scaling")
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        if 'price' in numeric_columns:
            numeric_columns.remove('price')  # Don't scale target variable
        
        selected_features = st.multiselect("Select Features to Scale:", numeric_columns)
        scaling_method = st.selectbox("Scaling Method:", ["standard", "minmax"])
        
        if selected_features and st.button("Apply Scaling"):
            # Before scaling
            original_stats = data[selected_features].describe()
            
            # Apply scaling
            scaled_data = processor.scale_features(method=scaling_method, columns=selected_features)
            scaled_stats = scaled_data[selected_features].describe()
            
            # Show comparison
            st.subheader("Before vs After Scaling")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Original Data:**")
                st.dataframe(original_stats)
            
            with col2:
                st.write("**Scaled Data:**")
                st.dataframe(scaled_stats)
            
            # Visualization
            if len(selected_features) >= 1:
                feature = selected_features[0]
                
                fig = make_subplots(rows=1, cols=2, 
                                  subplot_titles=["Original", "Scaled"])
                
                fig.add_trace(
                    go.Histogram(x=data[feature], name="Original"),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Histogram(x=scaled_data[feature], name="Scaled"),
                    row=1, col=2
                )
                
                fig.update_layout(title=f"Distribution Comparison: {feature}")
                st.plotly_chart(fig, use_container_width=True)
    
    elif preprocess_type == "Categorical Encoding":
        st.subheader("Categorical Variable Encoding")
        
        categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
        
        if categorical_columns:
            selected_cat_features = st.multiselect("Select Categorical Features:", categorical_columns)
            encoding_method = st.selectbox("Encoding Method:", ["onehot", "label"])
            
            if selected_cat_features and st.button("Apply Encoding"):
                encoded_data = processor.encode_categorical_variables(
                    method=encoding_method, 
                    columns=selected_cat_features
                )
                
                st.success(f"Applied {encoding_method} encoding to selected features!")
                
                # Show shape change
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Original Shape", f"{data.shape[0]} × {data.shape[1]}")
                with col2:
                    st.metric("Encoded Shape", f"{encoded_data.shape[0]} × {encoded_data.shape[1]}")
                
                # Show new columns created
                if encoding_method == "onehot":
                    new_columns = [col for col in encoded_data.columns if col not in data.columns]
                    if new_columns:
                        st.write("**New Columns Created:**")
                        for col in new_columns[:10]:  # Show first 10
                            st.write(f"• {col}")
                        if len(new_columns) > 10:
                            st.write(f"... and {len(new_columns) - 10} more")
        else:
            st.info("No categorical variables found in the dataset.")

def show_model_evaluation(data, processor):
    """Display model training and evaluation tools"""
    st.markdown('<h2 class="section-header">Model Training & Evaluation</h2>', unsafe_allow_html=True)
    
    # Model training section
    st.subheader("Model Training Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        target_column = st.selectbox("Select Target Variable:", ['price'])
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
    with col2:
        random_state = st.number_input("Random State", value=42, min_value=1)
        
    if st.button("🚀 Train Models"):
        with st.spinner("Training models..."):
            try:
                # Prepare data
                X_train, X_test, y_train, y_test = processor.prepare_model_data(
                    target_column=target_column,
                    test_size=test_size,
                    random_state=random_state
                )
                
                # Train models
                evaluator = ModelEvaluator()
                results = evaluator.train_regression_models(X_train, X_test, y_train, y_test)
                
                # Store results in session state
                st.session_state['model_results'] = results
                st.session_state['model_data'] = (X_train, X_test, y_train, y_test)
                
                st.success("✅ Models trained successfully!")
                
            except Exception as e:
                st.error(f"Error training models: {str(e)}")
    
    # Display results if available
    if 'model_results' in st.session_state:
        results = st.session_state['model_results']
        X_train, X_test, y_train, y_test = st.session_state['model_data']
        
        st.subheader("Model Performance Comparison")
        
        # Create performance DataFrame
        performance_data = []
        for name, result in results.items():
            performance_data.append({
                'Model': name,
                'Train R²': f"{result['metrics']['train_r2']:.4f}",
                'Test R²': f"{result['metrics']['test_r2']:.4f}",
                'Test RMSE': f"${result['metrics']['test_rmse']:,.0f}",
                'Test MAE': f"${result['metrics']['test_mae']:,.0f}"
            })
        
        performance_df = pd.DataFrame(performance_data)
        st.dataframe(performance_df, use_container_width=True)
        
        # Model comparison chart
        evaluator = ModelEvaluator()
        comparison_fig = evaluator.create_model_comparison_plot(results)
        st.plotly_chart(comparison_fig, use_container_width=True)
        
        # Best model analysis
        best_model_name = max(results.keys(), 
                            key=lambda x: results[x]['metrics']['test_r2'])
        
        st.subheader(f"Best Model: {best_model_name}")
        
        # Residual analysis
        best_result = results[best_model_name]
        y_pred = best_result['test_predictions']
        
        residual_fig = evaluator.create_residual_plots(y_test, y_pred, 
                                                     title=f"Residual Analysis - {best_model_name}")
        st.plotly_chart(residual_fig, use_container_width=True)
        
        # Feature importance (if available)
        if hasattr(best_result['model'], 'feature_importances_'):
            importance_fig = evaluator.create_feature_importance_plot(
                best_result['model'], X_train.columns
            )
            if importance_fig:
                st.plotly_chart(importance_fig, use_container_width=True)

def show_interpretability_module(data, processor):
    """Display model interpretability tools"""
    st.markdown('<h2 class="section-header">🔬 Model Interpretability</h2>', unsafe_allow_html=True)
    
    # Check if models are trained
    if 'model_results' not in st.session_state:
        st.warning("⚠️ Please train models first in the Model Training & Evaluation section.")
        return
    
    results = st.session_state['model_results']
    X_train, X_test, y_train, y_test = st.session_state['model_data']
    
    # Model selection
    model_names = list(results.keys())
    selected_model_name = st.selectbox("Select Model for Interpretation:", model_names)
    
    if selected_model_name:
        selected_model = results[selected_model_name]['model']
        
        # Initialize interpreter
        interpreter = ModelInterpreter(selected_model, X_train, X_test)
        
        # Create tabs for different interpretability methods
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Feature Importance", 
            "🧠 SHAP Analysis", 
            "🔍 LIME Explanations", 
            "📊 Interactive Exploration"
        ])
        
        with tab1:
            st.subheader("Feature Importance Analysis")
            
            # Traditional feature importance (if available)
            if hasattr(selected_model, 'feature_importances_'):
                st.markdown("**🌳 Model Built-in Feature Importance:**")
                
                # Create feature importance DataFrame
                importance_df = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': selected_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Interactive bar plot
                fig = px.bar(
                    importance_df.head(10),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title=f'Top 10 Feature Importance - {selected_model_name}',
                    color='importance',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance table
                st.dataframe(importance_df.head(10), use_container_width=True)
            else:
                st.info("Traditional feature importance not available for this model type.")
        
        with tab2:
            st.subheader("SHAP (SHapley Additive exPlanations) Analysis")
            
            with st.spinner("Calculating SHAP values..."):
                shap_values, X_sample, explainer = interpreter.get_shap_values()
                
                if shap_values is not None:
                    # Enhanced SHAP analysis options with new interactive visualizations
                    shap_option = st.selectbox(
                        "Select SHAP Analysis:",
                        [
                            "🎯 Interactive Summary Dashboard",
                            "📊 Traditional Summary Plot",
                            "📈 Feature Importance Ranking",
                            "🌊 Waterfall Plot (Individual)",
                            "📉 Dependence Plot",
                            "⚡ Force Plot",
                            "🔀 Decision Plot (Multiple Paths)",
                            "🐝 Beeswarm Plot",
                            "🎭 Clustering Analysis"
                        ]
                    )
                    
                    if shap_option == "🎯 Interactive Summary Dashboard":
                        st.markdown("**🎯 Interactive SHAP Analysis Dashboard**")
                        st.info("Comprehensive view showing feature importance, distributions, impacts, and correlations")
                        
                        interactive_fig = interpreter.create_interactive_shap_summary(shap_values, X_sample)
                        if interactive_fig:
                            st.plotly_chart(interactive_fig, use_container_width=True)
                        
                        # Show insights
                        st.markdown("**💡 Key Insights:**")
                        mean_shap = np.abs(shap_values).mean(0)
                        top_3_features = sorted(zip(X_sample.columns, mean_shap), key=lambda x: x[1], reverse=True)[:3]
                        
                        col1, col2, col3 = st.columns(3)
                        for i, (feature, importance) in enumerate(top_3_features):
                            with [col1, col2, col3][i]:
                                st.metric(f"Top {i+1} Feature", feature, f"{importance:.4f}")
                    
                    elif shap_option == "📊 Traditional Summary Plot":
                        st.markdown("**📊 SHAP Summary Plot**")
                        shap_summary_fig = interpreter.create_shap_summary_plot(shap_values, X_sample)
                        st.plotly_chart(shap_summary_fig, use_container_width=True)
                        
                        st.info("""
                        **Understanding the SHAP Summary Plot:**
                        - Each point represents a house prediction
                        - X-axis: SHAP value (impact on prediction)
                        - Color: Feature value (red=high, blue=low)
                        - Features ranked by importance (top to bottom)
                        """)
                    
                    elif shap_option == "📈 Feature Importance Ranking":
                        st.markdown("**🎯 SHAP Feature Importance**")
                        
                        # Calculate mean absolute SHAP values
                        feature_importance = np.abs(shap_values).mean(0)
                        importance_df = pd.DataFrame({
                            'feature': X_sample.columns,
                            'mean_shap_value': feature_importance
                        }).sort_values('mean_shap_value', ascending=False)
                        
                        # Interactive bar plot
                        fig = px.bar(
                            importance_df,
                            x='mean_shap_value',
                            y='feature',
                            orientation='h',
                            title='SHAP Feature Importance (Mean |SHAP value|)',
                            color='mean_shap_value',
                            color_continuous_scale='plasma'
                        )
                        fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display top features
                        st.dataframe(importance_df, use_container_width=True)
                    
                    elif shap_option == "Waterfall Plot (Individual)":
                        st.markdown("**🌊 SHAP Waterfall Plot**")
                        
                        instance_idx = st.slider(
                            "Select Property Instance:", 
                            0, len(X_sample)-1, 0
                        )
                        
                        waterfall_fig = interpreter.create_shap_waterfall_plot(instance_idx, shap_values, X_sample)
                        if waterfall_fig:
                            st.plotly_chart(waterfall_fig, use_container_width=True)
                        
                        # Show property details
                        st.markdown("**🏠 Property Details:**")
                        property_details = X_sample.iloc[instance_idx].to_dict()
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            for i, (feature, value) in enumerate(property_details.items()):
                                if i < len(property_details) // 2:
                                    st.write(f"• **{feature}:** {value}")
                        
                        with col2:
                            for i, (feature, value) in enumerate(property_details.items()):
                                if i >= len(property_details) // 2:
                                    st.write(f"• **{feature}:** {value}")
                    
                    elif shap_option == "Dependence Plot":
                        st.markdown("**📈 SHAP Dependence Plot**")
                        
                        feature_for_dependence = st.selectbox(
                            "Select Feature:", 
                            X_train.columns.tolist()
                        )
                        
                        dependence_fig = interpreter.create_shap_dependence_plot(
                            feature_for_dependence, shap_values, X_sample
                        )
                        if dependence_fig:
                            st.plotly_chart(dependence_fig, use_container_width=True)
                        
                        st.info(f"""
                        **Understanding the Dependence Plot for {feature_for_dependence}:**
                        - X-axis: {feature_for_dependence} values
                        - Y-axis: SHAP values (impact on prediction)
                        - Shows how {feature_for_dependence} affects model predictions
                        """)
                    
                    elif shap_option == "Force Plot":
                        st.markdown("**⚡ SHAP Force Plot**")
                        st.info("Force plots show how each feature pushes the prediction above or below the expected value.")
                        
                        instance_idx = st.slider(
                            "Select Instance for Force Plot:", 
                            0, len(X_sample)-1, 0
                        )
                        
                        # Create a simplified force plot using bar chart
                        instance_shap = shap_values[instance_idx]
                        feature_names = X_sample.columns
                        
                        force_df = pd.DataFrame({
                            'feature': feature_names,
                            'shap_value': instance_shap,
                            'feature_value': X_sample.iloc[instance_idx].values
                        })
                        force_df['impact'] = force_df['shap_value'].apply(lambda x: 'Positive' if x > 0 else 'Negative')
                        force_df = force_df.sort_values('shap_value', key=abs, ascending=False)
                        
                        fig = px.bar(
                            force_df.head(10),
                            x='shap_value',
                            y='feature',
                            orientation='h',
                            color='impact',
                            title=f'SHAP Force Plot - Property {instance_idx}',
                            color_discrete_map={'Positive': 'green', 'Negative': 'red'},
                            hover_data=['feature_value']
                        )
                        fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif shap_option == "🔀 Decision Plot (Multiple Paths)":
                        st.markdown("**🔀 SHAP Decision Plot - Multiple Prediction Paths**")
                        st.info("Shows how each feature contributes to the final prediction, step by step")
                        
                        decision_fig = interpreter.create_shap_decision_plot(shap_values, X_sample)
                        if decision_fig:
                            st.plotly_chart(decision_fig, use_container_width=True)
                        
                        st.markdown("""
                        **Understanding Decision Plots:**
                        - Each line represents a property's prediction path
                        - Y-axis shows cumulative prediction value
                        - X-axis shows decision steps (features)
                        - Starting point is the model's base prediction
                        """)
                    
                    elif shap_option == "🐝 Beeswarm Plot":
                        st.markdown("**🐝 SHAP Beeswarm Plot - Feature Impact Distribution**")
                        st.info("Alternative to summary plot showing feature value distributions and impacts")
                        
                        beeswarm_fig = interpreter.create_shap_beeswarm_plot(shap_values, X_sample)
                        if beeswarm_fig:
                            st.plotly_chart(beeswarm_fig, use_container_width=True)
                        
                        st.markdown("""
                        **Understanding Beeswarm Plots:**
                        - Each dot represents one property
                        - X-axis: SHAP value (impact on prediction)
                        - Y-axis: Features (ranked by importance)
                        - Color: Feature value (red=high, blue=low)
                        - Spread shows value distribution for each feature
                        """)
                    
                    elif shap_option == "🎭 Clustering Analysis":
                        st.markdown("**🎭 SHAP Clustering Analysis - Similar Prediction Patterns**")
                        st.info("Groups properties with similar SHAP value patterns to identify prediction archetypes")
                        
                        clustering_result = interpreter.create_shap_clustering_plot(shap_values, X_sample)
                        if clustering_result and clustering_result[0] is not None:
                            clustering_fig, cluster_labels = clustering_result
                            st.plotly_chart(clustering_fig, use_container_width=True)
                            
                            # Show cluster statistics
                            st.markdown("**📊 Cluster Analysis:**")
                            n_clusters = len(np.unique(cluster_labels))
                            
                            # Create cluster summary
                            cluster_summary = []
                            for cluster in range(n_clusters):
                                mask = cluster_labels == cluster
                                cluster_size = np.sum(mask)
                                cluster_predictions = [selected_model.predict(X_sample.iloc[[i]])[0] for i in range(len(X_sample)) if mask[i]]
                                avg_prediction = np.mean(cluster_predictions) if cluster_predictions else 0
                                
                                cluster_summary.append({
                                    'Cluster': f'Cluster {cluster + 1}',
                                    'Properties': cluster_size,
                                    'Avg Prediction': f'${avg_prediction:,.0f}',
                                    'Percentage': f'{cluster_size/len(cluster_labels)*100:.1f}%'
                                })
                            
                            cluster_df = pd.DataFrame(cluster_summary)
                            st.dataframe(cluster_df, use_container_width=True)
                        else:
                            st.warning("Clustering analysis requires additional libraries. Showing feature importance instead.")
                            fallback_fig = interpreter.create_feature_importance_fallback()
                            if fallback_fig:
                                st.plotly_chart(fallback_fig, use_container_width=True)
                else:
                    st.error("SHAP analysis requires the 'shap' library. Please install it to use this feature.")
        
        with tab3:
            st.subheader("LIME (Local Interpretable Model-agnostic Explanations)")
            
            instance_idx = st.slider("Select Property Instance for LIME:", 0, len(X_test)-1, 0)
            
            with st.spinner("Generating LIME explanation..."):
                lime_fig = interpreter.create_lime_plot(instance_idx)
                if lime_fig:
                    st.plotly_chart(lime_fig, use_container_width=True)
                    
                    # Show instance details
                    st.markdown("**🏠 Property Details for LIME Analysis:**")
                    instance_details = X_test.iloc[instance_idx].to_dict()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        for i, (feature, value) in enumerate(instance_details.items()):
                            if i < len(instance_details) // 2:
                                st.write(f"• **{feature}:** {value}")
                    
                    with col2:
                        for i, (feature, value) in enumerate(instance_details.items()):
                            if i >= len(instance_details) // 2:
                                st.write(f"• **{feature}:** {value}")
                    
                    # Actual vs Predicted
                    actual_price = y_test.iloc[instance_idx]
                    predicted_price = selected_model.predict(X_test.iloc[[instance_idx]])[0]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Actual Price", f"${actual_price:,.0f}")
                    with col2:
                        st.metric("Predicted Price", f"${predicted_price:,.0f}")
                    with col3:
                        error = abs(actual_price - predicted_price)
                        st.metric("Absolute Error", f"${error:,.0f}")
                else:
                    st.error("LIME analysis requires the 'lime' library. Please install it to use this feature.")
        
        with tab4:
            st.subheader("Interactive Exploration")
            
            # Comparison between SHAP and traditional importance
            if hasattr(selected_model, 'feature_importances_'):
                st.markdown("**🔄 SHAP vs Traditional Feature Importance**")
                
                with st.spinner("Computing comparison..."):
                    shap_values, X_sample, _ = interpreter.get_shap_values()
                    
                    if shap_values is not None:
                        # SHAP importance
                        shap_importance = np.abs(shap_values).mean(0)
                        shap_df = pd.DataFrame({
                            'feature': X_sample.columns,
                            'shap_importance': shap_importance / shap_importance.max()
                        })
                        
                        # Traditional importance
                        trad_importance = selected_model.feature_importances_
                        trad_df = pd.DataFrame({
                            'feature': X_train.columns,
                            'traditional_importance': trad_importance / trad_importance.max()
                        })
                        
                        # Merge and compare
                        comparison_df = pd.merge(shap_df, trad_df, on='feature')
                        
                        # Scatter plot comparison
                        fig = px.scatter(
                            comparison_df,
                            x='traditional_importance',
                            y='shap_importance',
                            text='feature',
                            title='SHAP vs Traditional Feature Importance',
                            hover_data=['feature']
                        )
                        fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, 
                                     line=dict(dash='dash', color='red'))
                        fig.update_traces(textposition='top center')
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Correlation
                        correlation = comparison_df['shap_importance'].corr(comparison_df['traditional_importance'])
                        st.metric("Correlation between methods", f"{correlation:.3f}")
            
            # Feature interaction exploration
            st.markdown("**🔗 Feature Interaction Analysis**")
            
            col1, col2 = st.columns(2)
            with col1:
                feature1 = st.selectbox("Select First Feature:", X_train.columns.tolist(), key="int_feat1")
            with col2:
                feature2 = st.selectbox("Select Second Feature:", X_train.columns.tolist(), key="int_feat2")
            
            if feature1 != feature2:
                with st.spinner("Analyzing feature interaction..."):
                    interaction_fig = interpreter.create_feature_interaction_plot(feature1, feature2)
                    if interaction_fig:
                        st.plotly_chart(interaction_fig, use_container_width=True)
                    else:
                        st.info("Feature interaction plots require scikit-learn >= 0.22.")
            
            # Model behavior summary
            st.markdown("**📋 Model Behavior Summary**")
            
            model_perf = results[selected_model_name]['metrics']
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Test R²", f"{model_perf['test_r2']:.4f}")
            with col2:
                st.metric("Test RMSE", f"${model_perf['test_rmse']:,.0f}")
            with col3:
                st.metric("Test MAE", f"${model_perf['test_mae']:,.0f}")
            with col4:
                overfitting = model_perf['train_r2'] - model_perf['test_r2']
                st.metric("Overfitting", f"{overfitting:.4f}")
            
            # Interpretability summary
            st.markdown("**🎯 Key Insights:**")
            
            if hasattr(selected_model, 'feature_importances_'):
                top_features = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': selected_model.feature_importances_
                }).sort_values('importance', ascending=False).head(3)
                
                st.write("**Top 3 Most Important Features:**")
                for i, (_, row) in enumerate(top_features.iterrows(), 1):
                    st.write(f"{i}. **{row['feature']}**: {row['importance']:.3f}")
            
            st.info("""
            **💡 Tips for Model Interpretability:**
            - Use SHAP for consistent, theoretically-grounded explanations
            - Use LIME for model-agnostic local explanations
            - Compare different explanation methods for validation
            - Focus on the most important features for business decisions
            """)

def show_interactive_features(data, processor):
    """Display interactive features and custom analysis"""
    st.markdown('<h2 class="section-header">Interactive Features</h2>', unsafe_allow_html=True)
    
    # Interactive data filtering
    st.subheader("🔍 Interactive Data Filtering")
    
    # Create filters
    col1, col2 = st.columns(2)
    
    with col1:
        # Price range filter
        price_min, price_max = st.slider(
            "Price Range",
            min_value=int(data['price'].min()),
            max_value=int(data['price'].max()),
            value=(int(data['price'].min()), int(data['price'].max())),
            format="$%d"
        )
        
        # Bedrooms filter
        bedroom_options = sorted(data['bedrooms'].unique())
        selected_bedrooms = st.multiselect("Bedrooms", bedroom_options, default=bedroom_options)
    
    with col2:
        # Area filter
        area_min, area_max = st.slider(
            "Area Range (Sq Ft)",
            min_value=int(data['area'].min()),
            max_value=int(data['area'].max()),
            value=(int(data['area'].min()), int(data['area'].max()))
        )
        
        # Furnishing status filter
        if 'furnishingstatus' in data.columns:
            furnishing_options = data['furnishingstatus'].unique().tolist()
            selected_furnishing = st.multiselect("Furnishing Status", furnishing_options, default=furnishing_options)
    
    # Apply filters
    filtered_data = data[
        (data['price'] >= price_min) & 
        (data['price'] <= price_max) &
        (data['bedrooms'].isin(selected_bedrooms)) &
        (data['area'] >= area_min) &
        (data['area'] <= area_max)
    ]
    
    if 'furnishingstatus' in data.columns:
        filtered_data = filtered_data[filtered_data['furnishingstatus'].isin(selected_furnishing)]
    
    # Show filtered results
    st.subheader(f"Filtered Results ({len(filtered_data)} properties)")
    
    if len(filtered_data) > 0:
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Price", f"${filtered_data['price'].mean():,.0f}")
        with col2:
            st.metric("Median Price", f"${filtered_data['price'].median():,.0f}")
        with col3:
            st.metric("Avg Area", f"{filtered_data['area'].mean():.0f}")
        with col4:
            st.metric("Properties", len(filtered_data))
        
        # Interactive scatter plot
        st.subheader("Interactive Scatter Plot")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            x_axis = st.selectbox("X-axis:", data.select_dtypes(include=[np.number]).columns.tolist())
        with col2:
            y_axis = st.selectbox("Y-axis:", data.select_dtypes(include=[np.number]).columns.tolist(), index=1)
        with col3:
            color_by = st.selectbox("Color by:", ['None'] + data.select_dtypes(include=['object']).columns.tolist())
        
        # Create scatter plot
        if color_by == 'None':
            color_by = None
        
        fig = px.scatter(
            filtered_data,
            x=x_axis,
            y=y_axis,
            color=color_by,
            hover_data=['price'],
            title=f"{y_axis} vs {x_axis}"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        if st.checkbox("Show filtered data table"):
            st.dataframe(filtered_data, use_container_width=True)
    
    else:
        st.warning("No properties match the selected filters. Please adjust your criteria.")
    
    # Custom dataset upload
    st.subheader("📁 Custom Dataset Upload")
    
    uploaded_file = st.file_uploader("Upload your own housing dataset (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            custom_data = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(custom_data)} records with {len(custom_data.columns)} columns")
            
            # Show preview
            st.subheader("Custom Dataset Preview")
            st.dataframe(custom_data.head(), use_container_width=True)
            
            # Basic analysis of custom data
            if st.button("Analyze Custom Dataset"):
                custom_visualizer = EDAVisualizer(custom_data)
                
                # Summary statistics
                st.subheader("Summary Statistics")
                st.dataframe(custom_data.describe(), use_container_width=True)
                
                # Missing data analysis
                missing_fig = custom_visualizer.create_missing_data_plot()
                st.plotly_chart(missing_fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error loading custom dataset: {str(e)}")

def show_price_prediction(data, processor):
    """Display price prediction interface"""
    st.markdown('<h2 class="section-header">🏠 Price Prediction</h2>', unsafe_allow_html=True)
    
    # Check if models are trained
    if 'model_results' not in st.session_state:
        st.warning("⚠️ Please train models first in the Model Training & Evaluation section.")
        st.info("Go to the 'Model Training & Evaluation' page and click 'Train Models' to enable predictions.")
        return
    
    results = st.session_state['model_results']
    X_train, X_test, y_train, y_test = st.session_state['model_data']
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['metrics']['test_r2'])
    best_model = results[best_model_name]['model']
    
    st.success(f"🎯 Using best performing model: **{best_model_name}** (R² = {results[best_model_name]['metrics']['test_r2']:.4f})")
    
    # Create two tabs: Manual Input and Bulk Prediction
    tab1, tab2 = st.tabs(["🏠 Single Property Prediction", "📊 Bulk Prediction"])
    
    with tab1:
        st.subheader("Enter Property Details")
        
        # Create input form for all features
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🏠 Basic Information**")
            area = st.number_input("Area (sq ft)", min_value=1000, max_value=20000, value=7500, step=100)
            bedrooms = st.selectbox("Bedrooms", options=[1, 2, 3, 4, 5, 6], index=2)
            bathrooms = st.selectbox("Bathrooms", options=[1, 2, 3, 4, 5], index=1)
            stories = st.selectbox("Stories", options=[1, 2, 3, 4], index=1)
            parking = st.selectbox("Parking Spaces", options=[0, 1, 2, 3, 4], index=2)
        
        with col2:
            st.markdown("**🏛️ Property Features**")
            mainroad = st.selectbox("Main Road Access", options=["yes", "no"], index=0)
            guestroom = st.selectbox("Guest Room", options=["yes", "no"], index=1)
            basement = st.selectbox("Basement", options=["yes", "no"], index=1)
            hotwaterheating = st.selectbox("Hot Water Heating", options=["yes", "no"], index=1)
        
        with col3:
            st.markdown("**✨ Premium Features**")
            airconditioning = st.selectbox("Air Conditioning", options=["yes", "no"], index=0)
            prefarea = st.selectbox("Preferred Area", options=["yes", "no"], index=0)
            furnishingstatus = st.selectbox("Furnishing Status", 
                                          options=["furnished", "semi-furnished", "unfurnished"], 
                                          index=0)
        
        # Predict button
        if st.button("🔮 Predict Price", type="primary"):
            try:
                # Create input data
                input_data = pd.DataFrame({
                    'area': [area],
                    'bedrooms': [bedrooms],
                    'bathrooms': [bathrooms],
                    'stories': [stories],
                    'parking': [parking],
                    'mainroad': [mainroad],
                    'guestroom': [guestroom],
                    'basement': [basement],
                    'hotwaterheating': [hotwaterheating],
                    'airconditioning': [airconditioning],
                    'prefarea': [prefarea],
                    'furnishingstatus': [furnishingstatus]
                })
                
                # Process the input data (encode categorical variables)
                processed_input = processor.process_prediction_input(input_data)
                
                # Make prediction
                predicted_price = best_model.predict(processed_input)[0]
                
                # Display prediction with styling
                st.markdown("---")
                st.markdown("### 🎯 Prediction Results")
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(90deg, #4CAF50, #45a049);
                        padding: 20px;
                        border-radius: 10px;
                        text-align: center;
                        color: white;
                        font-size: 24px;
                        font-weight: bold;
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    ">
                        Predicted Price: ${predicted_price:,.0f}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Add confidence and explanation
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 Model Performance:**")
                    st.write(f"• Model: {best_model_name}")
                    st.write(f"• R² Score: {results[best_model_name]['metrics']['test_r2']:.4f}")
                    st.write(f"• RMSE: ${results[best_model_name]['metrics']['test_rmse']:,.0f}")
                
                with col2:
                    st.markdown("**🏠 Your Property Summary:**")
                    st.write(f"• Area: {area:,} sq ft")
                    st.write(f"• Bedrooms: {bedrooms}")
                    st.write(f"• Bathrooms: {bathrooms}")
                    st.write(f"• Furnishing: {furnishingstatus}")
                
                # Compare to market
                st.markdown("---")
                st.markdown("**📈 Market Comparison:**")
                market_avg = data['price'].mean()
                market_median = data['price'].median()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    diff_avg = predicted_price - market_avg
                    color = "green" if diff_avg > 0 else "red"
                    st.markdown(f"vs Market Avg: <span style='color:{color}'>{diff_avg:+,.0f}</span>", unsafe_allow_html=True)
                
                with col2:
                    diff_median = predicted_price - market_median
                    color = "green" if diff_median > 0 else "red"
                    st.markdown(f"vs Market Median: <span style='color:{color}'>{diff_median:+,.0f}</span>", unsafe_allow_html=True)
                
                with col3:
                    percentile = (data['price'] < predicted_price).mean() * 100
                    st.write(f"Market Percentile: {percentile:.1f}%")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.write("Please ensure all fields are filled correctly.")
    
    with tab2:
        st.subheader("Bulk Property Prediction")
        
        # Sample data template
        st.markdown("**📝 Upload a CSV file with the following columns:**")
        st.code("area,bedrooms,bathrooms,stories,parking,mainroad,guestroom,basement,hotwaterheating,airconditioning,prefarea,furnishingstatus")
        
        # Download template
        if st.button("📥 Download Template CSV"):
            template_data = pd.DataFrame({
                'area': [7500, 8960, 6500],
                'bedrooms': [4, 3, 3],
                'bathrooms': [2, 2, 1],
                'stories': [2, 2, 1],
                'parking': [2, 3, 1],
                'mainroad': ['yes', 'yes', 'no'],
                'guestroom': ['no', 'no', 'yes'],
                'basement': ['yes', 'no', 'no'],
                'hotwaterheating': ['no', 'no', 'yes'],
                'airconditioning': ['yes', 'yes', 'no'],
                'prefarea': ['yes', 'no', 'no'],
                'furnishingstatus': ['furnished', 'semi-furnished', 'unfurnished']
            })
            csv = template_data.to_csv(index=False)
            st.download_button(
                label="Download",
                data=csv,
                file_name="property_template.csv",
                mime="text/csv"
            )
        
        # File upload
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                bulk_data = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(bulk_data)} properties")
                
                # Show preview
                st.subheader("Data Preview")
                st.dataframe(bulk_data.head(), use_container_width=True)
                
                if st.button("🔮 Predict All Prices"):
                    with st.spinner("Making predictions..."):
                        # Process the bulk data
                        processed_bulk = processor.process_prediction_input(bulk_data)
                        
                        # Make predictions
                        predictions = best_model.predict(processed_bulk)
                        
                        # Add predictions to original data
                        bulk_data['predicted_price'] = predictions
                        
                        # Display results
                        st.subheader("Prediction Results")
                        st.dataframe(bulk_data, use_container_width=True)
                        
                        # Summary statistics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Properties", len(predictions))
                        with col2:
                            st.metric("Avg Predicted Price", f"${predictions.mean():,.0f}")
                        with col3:
                            st.metric("Max Price", f"${predictions.max():,.0f}")
                        with col4:
                            st.metric("Min Price", f"${predictions.min():,.0f}")
                        
                        # Download results
                        csv = bulk_data.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name="predicted_prices.csv",
                            mime="text/csv"
                        )
                        
                        # Visualization
                        fig = px.histogram(
                            bulk_data, 
                            x='predicted_price',
                            title="Distribution of Predicted Prices",
                            nbins=20
                        )
                        fig.update_xaxis(title="Predicted Price")
                        fig.update_yaxis(title="Count")
                        st.plotly_chart(fig, use_container_width=True)
                        
            except Exception as e:
                st.error(f"Error processing bulk data: {str(e)}")

if __name__ == "__main__":
    main()