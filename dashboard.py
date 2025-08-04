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
        data = pd.read_csv('data/housing_data.csv')
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
    st.markdown('<h2 class="section-header">Model Interpretability</h2>', unsafe_allow_html=True)
    
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
        
        # Interpretation type selection
        interp_type = st.selectbox(
            "Select Interpretation Method:",
            [
                "Feature Importance",
                "SHAP Analysis", 
                "LIME Explanation",
                "Partial Dependence",
                "Feature Interactions"
            ]
        )
        
        if interp_type == "Feature Importance":
            st.subheader("Feature Importance Analysis")
            
            importance_fig = interpreter.create_feature_importance_fallback()
            if importance_fig:
                st.plotly_chart(importance_fig, use_container_width=True)
            else:
                st.info("Feature importance not available for this model type.")
        
        elif interp_type == "SHAP Analysis":
            st.subheader("SHAP (SHapley Additive exPlanations) Analysis")
            
            with st.spinner("Calculating SHAP values..."):
                shap_values, X_sample, explainer = interpreter.get_shap_values()
                
                if shap_values is not None:
                    # SHAP summary plot
                    shap_summary_fig = interpreter.create_shap_summary_plot(shap_values, X_sample)
                    st.plotly_chart(shap_summary_fig, use_container_width=True)
                    
                    # SHAP waterfall plot for specific instance
                    st.subheader("SHAP Waterfall Plot")
                    instance_idx = st.slider("Select Instance:", 0, len(X_sample)-1, 0)
                    
                    waterfall_fig = interpreter.create_shap_waterfall_plot(instance_idx, shap_values, X_sample)
                    if waterfall_fig:
                        st.plotly_chart(waterfall_fig, use_container_width=True)
                    
                    # SHAP dependence plot
                    st.subheader("SHAP Dependence Plot")
                    feature_for_dependence = st.selectbox("Select Feature:", X_train.columns.tolist())
                    
                    dependence_fig = interpreter.create_shap_dependence_plot(
                        feature_for_dependence, shap_values, X_sample
                    )
                    if dependence_fig:
                        st.plotly_chart(dependence_fig, use_container_width=True)
                else:
                    st.info("SHAP analysis requires the 'shap' library. Please install it to use this feature.")
        
        elif interp_type == "LIME Explanation":
            st.subheader("LIME (Local Interpretable Model-agnostic Explanations)")
            
            instance_idx = st.slider("Select Instance for LIME:", 0, len(X_test)-1, 0)
            
            with st.spinner("Generating LIME explanation..."):
                lime_fig = interpreter.create_lime_plot(instance_idx)
                if lime_fig:
                    st.plotly_chart(lime_fig, use_container_width=True)
                else:
                    st.info("LIME analysis requires the 'lime' library. Please install it to use this feature.")
        
        elif interp_type == "Partial Dependence":
            st.subheader("Partial Dependence Plots")
            
            feature_for_pdp = st.selectbox("Select Feature for PDP:", X_train.columns.tolist())
            
            with st.spinner("Calculating partial dependence..."):
                pdp_fig = interpreter.create_partial_dependence_plot(feature_for_pdp)
                if pdp_fig:
                    st.plotly_chart(pdp_fig, use_container_width=True)
                else:
                    st.info("Partial dependence plots require scikit-learn >= 0.22.")
        
        elif interp_type == "Feature Interactions":
            st.subheader("Feature Interaction Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                feature1 = st.selectbox("Select First Feature:", X_train.columns.tolist(), key="feat1")
            with col2:
                feature2 = st.selectbox("Select Second Feature:", X_train.columns.tolist(), key="feat2")
            
            if feature1 != feature2:
                with st.spinner("Analyzing feature interaction..."):
                    interaction_fig = interpreter.create_feature_interaction_plot(feature1, feature2)
                    if interaction_fig:
                        st.plotly_chart(interaction_fig, use_container_width=True)
                    else:
                        st.info("Feature interaction plots require scikit-learn >= 0.22.")

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

if __name__ == "__main__":
    main()