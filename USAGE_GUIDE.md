# Housing Price ML Dashboard - Usage Guide

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Interactive Dashboard**
   ```bash
   streamlit run dashboard.py
   ```

3. **Open Jupyter Notebook for Detailed Analysis**
   ```bash
   jupyter notebook notebooks/housing_analysis.ipynb
   ```

## Dashboard Features

### 📊 Data Overview
- Dataset summary with key metrics
- Data types and missing value analysis
- Interactive data preview tables

### 🔍 Exploratory Data Analysis
- **Target Variable Analysis**: Price distribution with histograms and box plots
- **Feature Distributions**: Interactive histograms and scatter plots for all variables
- **Correlation Analysis**: Comprehensive correlation matrix with heatmaps
- **Outlier Detection**: IQR and Z-score based outlier identification
- **Categorical Analysis**: Distribution analysis for categorical variables

### 🔧 Data Preprocessing
- **Missing Value Handling**: Multiple imputation strategies (mean, median, mode)
- **Feature Scaling**: StandardScaler and MinMaxScaler options
- **Categorical Encoding**: One-hot and label encoding methods
- **Before/After Comparisons**: Visual comparison of preprocessing effects

### 🤖 Model Training & Evaluation
- **Multiple Algorithms**: Linear Regression, Random Forest, Gradient Boosting
- **Performance Metrics**: R², RMSE, MAE for comprehensive evaluation
- **Residual Analysis**: Detailed residual plots and Q-Q plots
- **Model Comparison**: Side-by-side performance comparison charts
- **Feature Importance**: Automated feature importance visualization

### 🔬 Model Interpretability
- **SHAP Analysis**: Global and local explanations with summary plots
- **LIME Explanations**: Local interpretable explanations for individual predictions
- **Partial Dependence Plots**: Understanding feature effects on predictions
- **Feature Interactions**: 2D interaction effect visualizations

### 📈 Interactive Features
- **Dynamic Filtering**: Real-time data filtering with multiple criteria
- **Custom Visualizations**: User-selectable chart types and variables
- **Dataset Upload**: Support for custom CSV datasets
- **Export Capabilities**: Download processed data and visualizations

## Dataset Information

The project includes a synthetic housing dataset with 2,000 records and 16 features:

**Property Features:**
- Bedrooms, Bathrooms, Square_Feet, Lot_Size
- Year_Built, Age, Property_Type
- Garage_Spaces, Has_Pool, Has_Fireplace, Has_AC

**Location Features:**
- Neighborhood, Distance_to_School, Distance_to_Shopping, Distance_to_Highway

**Target Variable:**
- Price (in USD)

## Technical Architecture

### Backend Components
- **data_processing.py**: Data loading, cleaning, and preprocessing utilities
- **eda_module.py**: Exploratory data analysis and visualization functions
- **model_evaluation.py**: Machine learning model training and evaluation
- **interpretability.py**: Model explanation and interpretability tools

### Frontend
- **Streamlit Dashboard**: Interactive web-based interface
- **Plotly Visualizations**: Interactive charts and graphs
- **Responsive Design**: Works on desktop and mobile devices

### Analysis Notebook
- **housing_analysis.ipynb**: Comprehensive step-by-step analysis
- Detailed explanations and insights
- Reproducible research workflow

## Key Insights from Analysis

1. **Price Drivers**: Square footage is the strongest predictor of housing prices
2. **Location Premium**: Downtown and Uptown neighborhoods command highest prices
3. **Property Age Impact**: Newer homes generally have higher values
4. **Amenity Value**: Pools and garages significantly increase property values
5. **Model Performance**: Random Forest achieves ~90%+ R² score for price prediction

## Advanced Usage

### Custom Dataset Integration
1. Upload CSV file using the dashboard interface
2. Ensure target variable is clearly labeled
3. Use preprocessing tools to clean and prepare data
4. Train models and analyze results

### Model Interpretability
1. Train models using the dashboard
2. Navigate to Model Interpretability section
3. Select interpretation method (SHAP, LIME, PDP)
4. Analyze feature importance and interactions

### Extending Functionality
The modular design allows easy extension:
- Add new machine learning algorithms in `model_evaluation.py`
- Create custom visualizations in `eda_module.py`
- Implement new interpretability methods in `interpretability.py`

## Troubleshooting

**Common Issues:**
1. **Missing Dependencies**: Run `pip install -r requirements.txt`
2. **Data Loading Errors**: Ensure CSV file format is correct
3. **Model Training Failures**: Check for missing values in target variable
4. **Memory Issues**: Reduce dataset size or use sampling for large datasets

**Performance Tips:**
- Use data sampling for large datasets (>10,000 records)
- Enable caching in Streamlit for faster repeated operations
- Close unused browser tabs to free memory

## Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.