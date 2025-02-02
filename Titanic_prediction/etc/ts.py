import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import os
import joblib
import pickle

# Load the Dataset
file_path = r'C:\Users\Dell\Desktop\titanic_prediction\data\raw\titanic.csv'
df = pd.read_csv(file_path)

# Feature Engineering
def create_family_size(df):
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    return df

def extract_title(df):
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    return df

# Apply feature engineering
df = create_family_size(df)
df = extract_title(df)

# Prepare features and target
X = df.drop(['Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
y = df['Survived']

# Split the data first
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing (same as before)
numeric_features = ['Age', 'Fare', 'FamilySize']
categorical_features = ['Sex', 'Embarked', 'Title', 'Pclass']

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
                  ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)])

# Create a pipeline with preprocessor and classifier
rf_pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))])

# Hyperparameter tuning with RandomizedSearchCV
param_dist = {
    'classifier__n_estimators': [100, 150, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2],
    'classifier__max_features': ['sqrt', 'log2'],  # Replace 'auto' with 'sqrt' or 'log2'
    'classifier__bootstrap': [True, False]
}

randomized_search = RandomizedSearchCV(rf_pipeline, param_dist, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)

# Fit the randomized search
randomized_search.fit(X_train, y_train)

# Best model
best_rf_model = randomized_search.best_estimator_

# Predictions
y_pred = best_rf_model.predict(X_test)

# Evaluation metrics
print("Best Parameters:", randomized_search.best_params_)
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nROC AUC Score:", roc_auc_score(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show(block=False)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Fit the preprocessor to X_train to get the transformed feature names
X_train_transformed = best_rf_model.named_steps['preprocessor'].transform(X_train)

# Get feature names after one-hot encoding
ohe_feature_names = best_rf_model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features)

# Combine the transformed feature names
transformed_features = np.concatenate([numeric_features, ohe_feature_names])

# Get the feature importances from the RandomForest model
feature_importances = pd.DataFrame({
    'Feature': transformed_features,
    'Importance': best_rf_model.named_steps['classifier'].feature_importances_
}).sort_values(by='Importance', ascending=False)

# Feature importance plot
plt.figure(figsize=(8, 6))
sns.barplot(data=feature_importances, x='Importance', y='Feature', palette='viridis',hue='Importance')
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')

# Save the model
model_path = r'C:\Users\Dell\Desktop\titanic_prediction\models\titanic_improved_rf_model.pkl'
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(best_rf_model, model_path)
print(f"\nImproved Random Forest model saved at {model_path}")

plt.show()  # Display the feature importance plot