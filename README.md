# Housing Price Data Analysis & ML Dashboard

A comprehensive, interactive dashboard for machine learning practitioners and data scientists to visualize and interpret various stages of a machine learning workflow using housing price data.

## Features

### 📊 Exploratory Data Analysis (EDA) Module
- Interactive histograms, scatter plots, box plots, and pair plots
- Correlation matrices and heatmaps
- Outlier detection using visual aids
- Missing data visualization and pattern analysis

### 🔧 Data Preprocessing Visualization
- Before/after normalization and standardization comparisons
- Categorical encoding methods visualization
- Feature selection and extraction techniques
- Data transformation insights

### 📈 Model Evaluation Module
- **Classification**: ROC curves, confusion matrices, precision-recall curves
- **Regression**: Residual plots, Q-Q plots, prediction vs actual plots
- Interactive feature importance visualizations
- Cross-validation results and metrics

### 🔍 Model Interpretability Module
- SHAP (SHapley Additive exPlanations) interpretations
- LIME (Local Interpretable Model-agnostic Explanations)
- Partial dependence plots
- Feature interaction visualizations

### 🎯 Interactive Features
- Dynamic data filtering and zooming
- Real-time visualization type changes
- Custom dataset upload capability
- Model comparison tools

## Dataset

This project uses the [Housing Price Data](https://www.kaggle.com/datasets/saurabhbadole/housing-price-data) from Kaggle, which contains comprehensive information about housing features and prices.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AhmedAl-Mahdi/Housing-Price-Data.git
cd Housing-Price-Data
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Jupyter notebook for detailed analysis:
```bash
jupyter notebook housing_analysis.ipynb
```

4. Launch the interactive dashboard:
```bash
streamlit run dashboard.py
```

## Project Structure

```
Housing-Price-Data/
├── data/                          # Dataset files
├── notebooks/                     # Jupyter notebooks
│   └── housing_analysis.ipynb     # Comprehensive EDA and analysis
├── src/                           # Source code modules
│   ├── data_processing.py         # Data loading and preprocessing
│   ├── eda_module.py              # EDA visualization functions
│   ├── model_evaluation.py       # Model evaluation utilities
│   └── interpretability.py       # Model interpretability tools
├── dashboard.py                   # Main Streamlit dashboard
├── requirements.txt               # Project dependencies
└── README.md                      # This file
```

## Usage

### Jupyter Notebook Analysis
The `housing_analysis.ipynb` notebook provides:
- Detailed exploratory data analysis
- Step-by-step data preprocessing
- Model training and evaluation
- Insights and findings discussion

### Interactive Dashboard
The Streamlit dashboard offers:
- Real-time data exploration
- Interactive model training
- Dynamic visualization updates
- Model interpretability tools

## Technology Stack

- **Backend**: Python, Pandas, NumPy, Scikit-learn, PyTorch
- **Visualization**: Plotly, Seaborn, Matplotlib
- **Dashboard**: Streamlit
- **Model Interpretability**: SHAP, LIME
- **Analysis**: Jupyter Notebooks

## Contributing

Feel free to contribute by:
- Adding new visualization types
- Implementing additional ML models
- Enhancing interpretability features
- Improving dashboard UX/UI

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
