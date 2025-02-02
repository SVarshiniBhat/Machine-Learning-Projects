import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.inspection import PartialDependenceDisplay

# Load and display initial data
file_path = 'C:/Users/Dell/Sales Prediction/data/raw/advertising.csv'
data = pd.read_csv(file_path)
print("Initial Data:")
print(data.head())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Display Data Info and Statistics
print("\nData Info and Statistics:")
print(data.info())
print(data.describe())

# Save cleaned data (assuming no cleaning is needed here)
cleaned_data_path = 'C:/Users/Dell/Sales Prediction/data/processed/advertising_cleaned.csv'
data.to_csv(cleaned_data_path, index=False)
print(f"\nCleaned data saved to: {cleaned_data_path}")

# Splitting the data into features and target
X = data.drop(columns=['Sales'])
y = data['Sales']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the RandomForest model with hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 7, 10],
    'min_samples_split': [2, 5, 10]
}
model = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters and model
best_model = grid_search.best_estimator_
print(f"\nBest Parameters: {grid_search.best_params_}")

# Predictions
y_pred = best_model.predict(X_test)

# Evaluation Metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f"\nR² Score: {r2}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}%")

# Visualization

# 1. Feature Importance from RandomForest
feat_importances = pd.Series(best_model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh', figsize=(10, 6))
plt.title('Top 5 Feature Importances')
plt.show()


# 2. Partial Dependence Plot
PartialDependenceDisplay.from_estimator(best_model, X_train, features=['TV', 'Radio'])
plt.suptitle('Partial Dependence Plots for TV and Radio')
plt.subplots_adjust(top=0.9)  # Adjust title position
plt.show()

# 3. Residual Plot
plt.figure(figsize=(10, 6))
sns.residplot(x=y_pred, y=y_test - y_pred, lowess=True, color='g', line_kws={'color': 'r'})
plt.title('Residual Plot')
plt.xlabel('Predicted Sales')
plt.ylabel('Residuals')
plt.show()

# 4. Actual vs Predicted Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, color='b')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Actual vs Predicted Sales')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.show()

# 5. Distribution of Residuals
plt.figure(figsize=(10, 6))
sns.histplot(y_test - y_pred, kde=True, color='purple')
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.show()

# 6. Pairplot for feature correlations
sns.pairplot(data, kind='scatter')
plt.title('Pairplot of Features')
plt.show()

# 7. Correlation Heatmap
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# 8. Grid Search Results for Hyperparameters
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results[['param_n_estimators', 'param_max_depth', 'param_min_samples_split', 'mean_test_score']].sort_values(by='mean_test_score', ascending=False).head(10)

# Save the trained model for future use
import joblib
model_path = 'C:/Users/Dell/Sales Prediction/models/random_forest_sales_model.pkl'
joblib.dump(best_model, model_path)
print(f"\nTrained model saved to: {model_path}")
