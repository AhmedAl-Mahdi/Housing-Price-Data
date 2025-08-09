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
</style>
""", unsafe_allow_html=True)

def load_data():
    """Load the housing dataset"""
    try:
        data = pd.read_csv('Housing_Price_Data.csv')
        return data
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'Housing_Price_Data.csv' is in the project directory.")
        return None

def main():
    """Main dashboard function"""
    # Header
    st.markdown('<h1 class="main-header">🏠 Housing Price Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    data = load_data()
    if data is None:
        return
    
    # Initialize processor
    processor = DataProcessor(data)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
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
        st.metric("Numeric Features", len(data.select_dtypes(include=[np.number]).columns))
    with col4:
        st.metric("Categorical Features", len(data.select_dtypes(include=['object']).columns))
    
    # Display first few rows
    st.subheader("Sample Data")
    st.dataframe(data.head(), use_container_width=True)
    
    # Data types and info
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Types")
        dtype_df = pd.DataFrame({
            'Column': data.columns,
            'Data Type': data.dtypes,
            'Non-Null Count': data.count(),
            'Null Count': data.isnull().sum()
        })
        st.dataframe(dtype_df, use_container_width=True)
    
    with col2:
        st.subheader("Summary Statistics")
        st.dataframe(data.describe(), use_container_width=True)
    
    # Missing data visualization
    if data.isnull().sum().sum() > 0:
        st.subheader("Missing Data Analysis")
        missing_data = data.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) > 0:
            fig = px.bar(
                x=missing_data.values,
                y=missing_data.index,
                orientation='h',
                title="Missing Values by Feature",
                labels={'x': 'Number of Missing Values', 'y': 'Features'}
            )
            st.plotly_chart(fig, use_container_width=True)

def show_eda_module(data, processor):
    """Display exploratory data analysis"""
    st.markdown('<h2 class="section-header">Exploratory Data Analysis</h2>', unsafe_allow_html=True)
    
    # Initialize visualizer
    visualizer = EDAVisualizer(data)
    
    # Analysis type selection
    analysis_type = st.selectbox(
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
        st.subheader("Target Variable (Price) Analysis")
        
        # Price distribution
        fig = px.histogram(data, x='price', nbins=30, title="Price Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        # Price statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Price", f"${data['price'].mean():,.0f}")
        with col2:
            st.metric("Median Price", f"${data['price'].median():,.0f}")
        with col3:
            st.metric("Min Price", f"${data['price'].min():,.0f}")
        with col4:
            st.metric("Max Price", f"${data['price'].max():,.0f}")
    
    elif analysis_type == "Feature Distributions":
        st.subheader("Feature Distributions")
        
        # Feature selection
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        if 'price' in numeric_columns:
            numeric_columns.remove('price')
        
        selected_feature = st.selectbox("Select Feature:", numeric_columns)
        
        if selected_feature:
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram
                fig = px.histogram(data, x=selected_feature, nbins=20, 
                                 title=f"Distribution of {selected_feature}")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot
                fig = px.box(data, y=selected_feature, title=f"Box Plot of {selected_feature}")
                st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Correlation Analysis":
        st.subheader("Correlation Analysis")
        
        # Correlation heatmap
        numeric_data = data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()
        
        fig = px.imshow(correlation_matrix, 
                       title="Feature Correlation Heatmap",
                       color_continuous_scale='RdBu_r',
                       aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
        
        # Top correlations with price
        if 'price' in correlation_matrix.columns:
            price_corr = correlation_matrix['price'].abs().sort_values(ascending=False)
            price_corr = price_corr[price_corr.index != 'price']  # Remove self-correlation
            
            st.subheader("Features Most Correlated with Price")
            fig = px.bar(y=price_corr.index[:10], x=price_corr.values[:10], 
                        orientation='h', title="Top 10 Features by Correlation with Price")
            st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Outlier Detection":
        st.subheader("Outlier Detection")
        
        # Feature selection for outlier analysis
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        selected_feature = st.selectbox("Select Feature for Outlier Analysis:", numeric_columns)
        
        if selected_feature:
            # Outlier detection method
            method = st.selectbox("Detection Method:", ["IQR", "Z-Score"])
            
            outlier_info = processor.detect_outliers(selected_feature, method.lower().replace('-', '_'))
            
            if outlier_info:
                # Display outlier statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Outliers Found", outlier_info['count'])
                with col2:
                    st.metric("Percentage", f"{outlier_info['percentage']:.1f}%")
                
                # Outlier visualization
                outlier_fig = visualizer.create_outlier_plot(selected_feature, method.lower().replace('-', '_'))
                st.plotly_chart(outlier_fig, use_container_width=True)
    
    elif analysis_type == "Categorical Analysis":
        st.subheader("Categorical Feature Analysis")
        
        # Get categorical columns
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
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Price by category
                    fig = px.box(data, x=selected_cat, y='price',
                               title=f"Price Distribution by {selected_cat}")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No categorical features found in the dataset.")

def show_preprocessing_module(data, processor):
    """Display data preprocessing options"""
    st.markdown('<h2 class="section-header">Data Preprocessing</h2>', unsafe_allow_html=True)
    
    # Preprocessing options
    preprocess_type = st.selectbox(
        "Select Preprocessing Task:",
        [
            "Missing Value Handling",
            "Feature Scaling", 
            "Encoding Categorical Variables",
            "Feature Engineering"
        ]
    )
    
    if preprocess_type == "Missing Value Handling":
        st.subheader("Missing Value Analysis")
        
        missing_summary = data.isnull().sum()
        missing_summary = missing_summary[missing_summary > 0]
        
        if len(missing_summary) == 0:
            st.success("✅ No missing values found in the dataset!")
        else:
            st.warning(f"Found missing values in {len(missing_summary)} columns")
            
            # Display missing value summary
            missing_df = pd.DataFrame({
                'Feature': missing_summary.index,
                'Missing Count': missing_summary.values,
                'Percentage': (missing_summary.values / len(data)) * 100
            })
            st.dataframe(missing_df, use_container_width=True)
            
            # Missing value visualization
            fig = px.bar(missing_df, x='Feature', y='Missing Count',
                        title="Missing Values by Feature")
            st.plotly_chart(fig, use_container_width=True)
    
    elif preprocess_type == "Feature Scaling":
        st.subheader("Feature Scaling")
        st.info("Feature scaling helps normalize the range of features for better model performance.")
        
        # Show current feature ranges
        numeric_data = data.select_dtypes(include=[np.number])
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Current Feature Ranges:**")
            ranges_df = pd.DataFrame({
                'Feature': numeric_data.columns,
                'Min': numeric_data.min(),
                'Max': numeric_data.max(),
                'Range': numeric_data.max() - numeric_data.min()
            })
            st.dataframe(ranges_df, use_container_width=True)
        
        with col2:
            # Scaling method selection
            scaling_method = st.selectbox(
                "Select Scaling Method:",
                ["StandardScaler", "MinMaxScaler", "RobustScaler"]
            )
            
            if st.button("Apply Scaling"):
                scaled_data = processor.scale_features(scaling_method.lower())
                st.success(f"✅ Applied {scaling_method} to numeric features")
                
                # Show scaled ranges
                st.write("**After Scaling:**")
                scaled_ranges = pd.DataFrame({
                    'Feature': scaled_data.columns,
                    'Min': scaled_data.min(),
                    'Max': scaled_data.max()
                })
                st.dataframe(scaled_ranges, use_container_width=True)
    
    elif preprocess_type == "Encoding Categorical Variables":
        st.subheader("Categorical Variable Encoding")
        
        categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
        
        if categorical_columns:
            st.write("**Categorical Features Found:**")
            for col in categorical_columns:
                unique_count = data[col].nunique()
                st.write(f"• **{col}**: {unique_count} unique values")
            
            # Encoding method selection
            encoding_method = st.selectbox(
                "Select Encoding Method:",
                ["One-Hot Encoding", "Label Encoding"]
            )
            
            if st.button("Apply Encoding"):
                if encoding_method == "One-Hot Encoding":
                    encoded_data = processor.encode_categorical_features("onehot")
                else:
                    encoded_data = processor.encode_categorical_features("label")
                
                st.success(f"✅ Applied {encoding_method}")
                st.write(f"Data shape after encoding: {encoded_data.shape}")
        else:
            st.info("No categorical features found in the dataset.")
    
    elif preprocess_type == "Feature Engineering":
        st.subheader("Feature Engineering")
        st.info("Create new features from existing ones to improve model performance.")
        
        # Example feature engineering options
        if st.checkbox("Create Price per Square Foot"):
            if 'area' in data.columns and 'price' in data.columns:
                data['price_per_sqft'] = data['price'] / data['area']
                st.success("✅ Created 'price_per_sqft' feature")
            else:
                st.error("Required columns 'area' and 'price' not found")
        
        if st.checkbox("Create Total Rooms"):
            if 'bedrooms' in data.columns and 'bathrooms' in data.columns:
                data['total_rooms'] = data['bedrooms'] + data['bathrooms']
                st.success("✅ Created 'total_rooms' feature")
            else:
                st.error("Required columns 'bedrooms' and 'bathrooms' not found")

def show_model_evaluation(data, processor):
    """Display model training and evaluation"""
    st.markdown('<h2 class="section-header">🤖 Model Training & Evaluation</h2>', unsafe_allow_html=True)
    
    st.info("Train and evaluate different machine learning models on the housing data.")
    
    # Model selection
    selected_models = st.multiselect(
        "Select Models to Train:",
        ["Random Forest", "Gradient Boosting", "Linear Regression", "Ridge Regression"],
        default=["Random Forest", "Linear Regression"]
    )
    
    if not selected_models:
        st.warning("Please select at least one model to train.")
        return
    
    # Feature selection for modeling
    feature_columns = [col for col in data.columns if col != 'price']
    selected_features = st.multiselect(
        "Select Features:",
        feature_columns,
        default=feature_columns[:5] if len(feature_columns) >= 5 else feature_columns
    )
    
    if not selected_features:
        st.warning("Please select at least one feature.")
        return
    
    if st.button("Train Models", type="primary"):
        with st.spinner("Training models..."):
            # Prepare data
            X = data[selected_features]
            y = data['price']
            
            # Handle categorical variables
            for col in X.select_dtypes(include=['object']).columns:
                X[col] = pd.Categorical(X[col]).codes
            
            # Train models
            evaluator = ModelEvaluator()
            results = evaluator.train_and_evaluate_models(X, y, selected_models)
            
            # Store results in session state
            st.session_state['model_results'] = results
            st.session_state['model_data'] = evaluator.get_train_test_data()
        
        st.success("✅ Models trained successfully!")
    
    # Show results if available
    if 'model_results' in st.session_state:
        results = st.session_state['model_results']
        
        st.subheader("Model Performance Comparison")
        
        # Create performance DataFrame
        performance_data = []
        for name, result in results.items():
            overfitting = result['metrics']['train_r2'] - result['metrics']['test_r2']
            performance_data.append({
                'Model': name,
                'Train R²': f"{result['metrics']['train_r2']:.4f}",
                'Test R²': f"{result['metrics']['test_r2']:.4f}",
                'Overfitting': f"{overfitting:.4f}",
                'Test RMSE': f"${result['metrics']['test_rmse']:,.0f}",
                'Test MAE': f"${result['metrics']['test_mae']:,.0f}"
            })
        
        performance_df = pd.DataFrame(performance_data)
        st.dataframe(performance_df, use_container_width=True)
        
        # Model comparison chart
        evaluator = ModelEvaluator()
        comparison_fig = evaluator.create_model_comparison_plot(results)
        st.plotly_chart(comparison_fig, use_container_width=True)

def show_interpretability_module(data, processor):
    """Display model interpretability tools"""
    st.markdown('<h2 class="section-header">🔬 Model Interpretability</h2>', unsafe_allow_html=True)
    
    # Check if models are trained
    if 'model_results' not in st.session_state:
        st.warning("⚠️ Please train models first in the Model Training & Evaluation section.")
        return
    
    results = st.session_state['model_results']
    
    # Model selection
    model_names = list(results.keys())
    selected_model_name = st.selectbox("Select Model for Interpretation:", model_names)
    
    if selected_model_name:
        selected_model = results[selected_model_name]['model']
        
        st.subheader(f"Interpreting: {selected_model_name}")
        
        # Feature importance (if available)
        if hasattr(selected_model, 'feature_importances_'):
            st.subheader("Feature Importance")
            
            # Get feature names from the last training
            if 'model_data' in st.session_state:
                X_train, X_test, y_train, y_test = st.session_state['model_data']
                
                importance_df = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': selected_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Interactive bar plot
                fig = px.bar(
                    importance_df,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title=f'Feature Importance - {selected_model_name}'
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance table
                st.dataframe(importance_df, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")

def show_price_prediction(data, processor):
    """Display price prediction interface"""
    st.markdown('<h2 class="section-header">🏠 Price Prediction</h2>', unsafe_allow_html=True)
    
    # Check if models are trained
    if 'model_results' not in st.session_state:
        st.warning("⚠️ Please train models first in the Model Training & Evaluation section.")
        return
    
    results = st.session_state['model_results']
    
    # Model selection
    model_names = list(results.keys())
    selected_model_name = st.selectbox("Select Model for Prediction:", model_names)
    
    if selected_model_name:
        selected_model = results[selected_model_name]['model']
        
        st.subheader("Enter Property Details")
        
        # Get the features used in training
        if 'model_data' in st.session_state:
            X_train, _, _, _ = st.session_state['model_data']
            feature_names = X_train.columns.tolist()
            
            # Create input form
            input_data = {}
            
            col1, col2 = st.columns(2)
            
            for i, feature in enumerate(feature_names):
                # Determine input type based on feature
                if feature in data.columns:
                    if data[feature].dtype in ['int64', 'float64']:
                        # Numeric input
                        min_val = float(data[feature].min())
                        max_val = float(data[feature].max())
                        default_val = float(data[feature].median())
                        
                        with col1 if i % 2 == 0 else col2:
                            input_data[feature] = st.number_input(
                                f"{feature.title()}:",
                                min_value=min_val,
                                max_value=max_val,
                                value=default_val
                            )
                    else:
                        # Categorical input
                        unique_values = data[feature].unique().tolist()
                        with col1 if i % 2 == 0 else col2:
                            input_data[feature] = st.selectbox(f"{feature.title()}:", unique_values)
                else:
                    # Engineered feature - use default
                    with col1 if i % 2 == 0 else col2:
                        input_data[feature] = st.number_input(f"{feature.title()}:", value=0.0)
            
            # Make prediction
            if st.button("Predict Price", type="primary"):
                try:
                    # Prepare input data
                    input_df = pd.DataFrame([input_data])
                    
                    # Handle categorical variables
                    for col in input_df.select_dtypes(include=['object']).columns:
                        if col in data.columns:
                            # Use the same encoding as original data
                            input_df[col] = pd.Categorical(input_df[col], 
                                                         categories=data[col].unique()).codes
                    
                    # Make prediction
                    prediction = selected_model.predict(input_df)[0]
                    
                    # Display result
                    st.success(f"🏠 Predicted Price: **${prediction:,.0f}**")
                    
                    # Show input summary
                    st.subheader("Input Summary")
                    summary_df = pd.DataFrame(list(input_data.items()), 
                                            columns=['Feature', 'Value'])
                    st.dataframe(summary_df, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")

def show_interactive_features(data, processor):
    """Display interactive features"""
    st.markdown('<h2 class="section-header">📈 Interactive Features</h2>', unsafe_allow_html=True)
    
    # Feature comparison
    st.subheader("Feature Comparison")
    
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("X-axis:", numeric_columns)
    with col2:
        y_axis = st.selectbox("Y-axis:", [col for col in numeric_columns if col != x_axis])
    
    if x_axis and y_axis:
        # Color by categorical variable
        categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
        if categorical_columns:
            color_by = st.selectbox("Color by:", ['None'] + categorical_columns)
            if color_by != 'None':
                fig = px.scatter(data, x=x_axis, y=y_axis, color=color_by,
                               title=f"{y_axis} vs {x_axis}")
            else:
                fig = px.scatter(data, x=x_axis, y=y_axis,
                               title=f"{y_axis} vs {x_axis}")
        else:
            fig = px.scatter(data, x=x_axis, y=y_axis,
                           title=f"{y_axis} vs {x_axis}")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Data filtering
    st.subheader("Data Filtering")
    
    # Price range filter
    if 'price' in data.columns:
        price_range = st.slider(
            "Price Range:",
            min_value=int(data['price'].min()),
            max_value=int(data['price'].max()),
            value=(int(data['price'].min()), int(data['price'].max()))
        )
        
        filtered_data = data[(data['price'] >= price_range[0]) & 
                           (data['price'] <= price_range[1])]
        
        st.write(f"Showing {len(filtered_data)} properties out of {len(data)}")
        st.dataframe(filtered_data.head(10), use_container_width=True)

if __name__ == "__main__":
    main()
