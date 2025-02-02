# Import Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
import joblib
import pickle

# Load the Dataset
file_path = r'C:\Users\Dell\Desktop\titanic_prediction\data\raw\titanic.csv'
df = pd.read_csv(file_path)

# Data Exploration
print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nDataset Description:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# Data Cleaning
df['Age'] = df['Age'].fillna(df['Age'].median())  # Fill missing Age with median
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])  # Fill missing Embarked with mode
df.drop('Cabin', axis=1, inplace=True)  # Drop Cabin as it has too many missing values
df.drop(['Name', 'Ticket'], axis=1, inplace=True)  # Drop irrelevant columns

# Verify if missing values are handled
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Save Cleaned Dataset
processed_path = r'C:\Users\Dell\Desktop\titanic_prediction\data\processed\cleaned_titanic_data.csv'
if not os.path.exists(os.path.dirname(processed_path)):
    os.makedirs(os.path.dirname(processed_path))  # Create directories if they don't exist
df.to_csv(processed_path, index=False)
print(f"\nCleaned dataset saved at {processed_path}")

# Encode Categorical Variables
label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])  # Encode 'Sex'
df['Embarked'] = label_encoder.fit_transform(df['Embarked'])  # Encode 'Embarked'

# Feature Selection and Data Splitting
X = df.drop('Survived', axis=1)  # Features
y = df['Survived']  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training - Logistic Regression
log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train, y_train)

# Predict and Evaluate - Logistic Regression
y_pred_log = log_model.predict(X_test)
log_accuracy = accuracy_score(y_test, y_pred_log)
print("\nLogistic Regression Accuracy:", log_accuracy)

# Model Training - Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict and Evaluate - Random Forest
y_pred_rf = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print("\nRandom Forest Accuracy:", rf_accuracy)

# Confusion Matrix Plot (First Plot)
plt.figure(figsize=(8, 6))  # Create a new figure for the confusion matrix
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show(block=False)  # Non-blocking to allow the second plot to open

# Classification Report
print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))

# Feature Importance Plot (Second Plot)
plt.figure(figsize=(8, 6))  # Create a new figure for the feature importance
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

sns.barplot(data=feature_importances, x='Importance', y='Feature', palette='viridis',hue='Importance')
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()  # Display the second plot

# Save the Model for Future Use
model_path = r'C:\Users\Dell\Desktop\titanic_prediction\models\titanic_rf_model.pkl'
if not os.path.exists(os.path.dirname(model_path)):
    os.makedirs(os.path.dirname(model_path))  # Create directories if they don't exist
joblib.dump(rf_model, model_path)
print(f"Random Forest model saved at {model_path}")

# Verify if the model is saved successfully
if os.path.exists(model_path):
    print(f"Model successfully saved at {model_path}")
else:
    print("Model not saved. Please check the directory permissions.")  