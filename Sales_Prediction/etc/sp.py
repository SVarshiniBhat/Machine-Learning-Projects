import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load Dataset
df = pd.read_csv("C:/Users/Dell/Sales Prediction/data/raw/advertising.csv")

# Data Cleaning & Preprocessing
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Exploratory Data Analysis
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")


sns.pairplot(df)


# Regression Plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.regplot(x=df['TV'], y=df['Sales'], ax=axes[0])
sns.regplot(x=df['Radio'], y=df['Sales'], ax=axes[1])
sns.regplot(x=df['Newspaper'], y=df['Sales'], ax=axes[2])
axes[0].set_title("Sales vs TV")
axes[1].set_title("Sales vs Radio")
axes[2].set_title("Sales vs Newspaper")


# Define Features and Target
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Models
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# Model Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'R²': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
    }

print("Linear Regression:", evaluate_model(lr, X_test_scaled, y_test))
print("Random Forest:", evaluate_model(rf, X_test_scaled, y_test))

# Hyperparameter Tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [ 'sqrt', 'log2',None]
}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='r2')
grid_search.fit(X_train_scaled, y_train)
best_rf = grid_search.best_estimator_
print("Best Random Forest Parameters:", grid_search.best_params_)

# Feature Importance Analysis
feature_importance = best_rf.feature_importances_
feature_names = X.columns
feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
feature_df['Importance'] /= feature_df['Importance'].sum()  # Normalize
feature_df = feature_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', hue='Feature', data=feature_df, legend=False, palette='coolwarm')
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance using Random Forest")
plt.show()

# SHAP Analysis
explainer = shap.Explainer(best_rf, X_train_scaled)
shap_values = explainer(X_test_scaled)
shap.summary_plot(shap_values, X_test)

# Budget Optimization (Adjusted)
total_budget = 100000
optimized_budget = {}

# Adjusted Budget Allocation based on feature importance
for feature, importance in zip(feature_names, feature_importance):
    optimized_value = total_budget * importance ** 2  # Emphasizing more influential features
    
    # Ensure the value is valid before adding to the dictionary
    if np.isnan(optimized_value) or np.isinf(optimized_value):
        print(f"Warning: Invalid value encountered for {feature}. Replacing with 0.")
        optimized_value = 0  # Replace invalid value with 0 or a reasonable default
    
    optimized_budget[feature] = np.float64(optimized_value)  # Safely cast to np.float64

# Normalize to ensure the sum is the total budget
normalized_budget = {key: (value / sum(optimized_budget.values())) * total_budget for key, value in optimized_budget.items()}
print("Optimized Budget Allocation:", normalized_budget)

# Predict Sales for Different Budgets
def predict_sales(model, budget):
    budget_df = pd.DataFrame([budget])
    budget_scaled = scaler.transform(budget_df)
    return model.predict(budget_scaled)[0]

sales_current = predict_sales(best_rf, {'TV': 33333, 'Radio': 33333, 'Newspaper': 33333})
sales_optimized = predict_sales(best_rf, normalized_budget)
print(f"Estimated Sales with Current Budget: {sales_current:.2f}")
print(f"Estimated Sales with Optimized Budget: {sales_optimized:.2f}")
print(f"Expected Sales Increase: {(sales_optimized - sales_current):.2f}")

# A/B Testing Simulation (Improved)
group_A = df[df['TV'] > df['TV'].median()]  # Group by a meaningful feature, such as TV spend
group_B = df[df['TV'] <= df['TV'].median()]
df['Test_Group'] = ['A' if i in group_A.index else 'B' for i in df.index]

# Print mean sales for A/B test groups
print(df.groupby('Test_Group')['Sales'].mean())
