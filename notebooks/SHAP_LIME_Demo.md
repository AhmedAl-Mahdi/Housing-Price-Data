# SHAP and LIME Demo - Housing Price Interpretability

## Quick Demo of Model Interpretability Features

This notebook demonstrates the key SHAP and LIME interpretability features added to the housing price analysis.

### Load Required Libraries


```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
```

### Load and Prepare Data


```python
# Load the housing dataset
df = pd.read_csv('data/housing_data.csv')

# Encode categorical variables (same as in main notebook)
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})

# One-hot encode furnishing status
furnishing_dummies = pd.get_dummies(df['furnishingstatus'], prefix='furnishing')
df = pd.concat([df, furnishing_dummies], axis=1)
df.drop('furnishingstatus', axis=1, inplace=True)

# Prepare features and target
X = df.drop('price', axis=1)
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Dataset: {X.shape[0]} houses with {X.shape[1]} features")
print(f"Features: {list(X.columns)}")
```

### Train Model


```python
# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Check performance
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Train R²: {train_score:.4f}")
print(f"Test R²: {test_score:.4f}")
```

### SHAP Analysis


```python
# Initialize SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# SHAP Summary Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False)
plt.title('SHAP Feature Importance Summary')
plt.tight_layout()
plt.show()

# Feature importance ranking
feature_importance = np.abs(shap_values).mean(0)
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("\\nTop 5 Most Important Features (SHAP):")
for i, row in importance_df.head(5).iterrows():
    print(f"{row['feature']}: {row['importance']:.3f}")
```

### LIME Analysis


```python
# Initialize LIME explainer
lime_explainer = LimeTabularExplainer(
    X_train.values,
    feature_names=X.columns,
    mode='regression',
    random_state=42
)

# Explain a specific prediction
house_idx = 0
explanation = lime_explainer.explain_instance(
    X_test.iloc[house_idx].values,
    model.predict,
    num_features=len(X.columns)
)

# Show explanation
print(f"\\nLIME Explanation for House {house_idx + 1}:")
print(f"Actual Price: ${y_test.iloc[house_idx]:,.0f}")
print(f"Predicted Price: ${model.predict(X_test.iloc[[house_idx]])[0]:,.0f}")
print("\\nFeature Contributions:")

for feature, contribution in explanation.as_list():
    direction = "increases" if contribution > 0 else "decreases"
    print(f"  {feature}: {contribution:+.0f} ({direction} price)")

# Show LIME plot
fig = explanation.as_pyplot_figure()
plt.title(f'LIME Explanation - House {house_idx + 1}')
plt.tight_layout()
plt.show()
```

### Individual Prediction Explanation


```python
# SHAP Waterfall Plot for the same house
import shap

# Create SHAP explanation object
expected_value = explainer.expected_value
shap_explanation = shap.Explanation(
    values=shap_values[house_idx],
    base_values=expected_value,
    data=X_test.iloc[house_idx].values,
    feature_names=X.columns.tolist()
)

# Create waterfall plot
plt.figure(figsize=(10, 6))
shap.waterfall_plot(shap_explanation, show=False)
plt.title(f'SHAP Waterfall Plot - House {house_idx + 1}')
plt.tight_layout()
plt.show()

print(f"\\nHouse {house_idx + 1} Characteristics:")
for col in X.columns:
    print(f"  {col}: {X_test.iloc[house_idx][col]}")
```

### Key Insights


```python
print("\\n" + "="*50)
print("KEY INTERPRETABILITY INSIGHTS")
print("="*50)

print("\\n🎯 Most Important Features:")
for i, row in importance_df.head(3).iterrows():
    print(f"  {i+1}. {row['feature']}: Strong predictor of house prices")

print("\\n🔍 SHAP Benefits:")
print("  • Global feature importance across all predictions")
print("  • Individual prediction breakdowns")
print("  • Consistent with game theory (Shapley values)")
print("  • Excellent visualizations")

print("\\n🔍 LIME Benefits:")
print("  • Model-agnostic explanations")
print("  • Local fidelity for individual predictions")
print("  • Intuitive for non-technical stakeholders")
print("  • Works with any ML model")

print("\\n💡 Business Applications:")
print("  • Justify property valuations to clients")
print("  • Identify key value drivers in real estate")
print("  • Build trust in automated pricing systems")
print("  • Meet regulatory transparency requirements")

print("\\n✅ The model is interpretable and ready for production use!")
```

## Summary

This demo showcases the power of combining SHAP and LIME for comprehensive model interpretability:

- **SHAP** provides global understanding of feature importance and consistent individual explanations
- **LIME** offers model-agnostic local explanations that are easy to understand
- **Together** they provide robust validation of model behavior and trustworthy explanations

The housing price model can now be deployed with confidence, knowing that every prediction can be explained clearly to stakeholders.