import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
 
# Load dataset
df = pd.read_csv(r"C:\Users\Dell\Desktop\titanic_prediction\data\raw\Titanic-Dataset.csv")
 
# Drop unnecessary columns
df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)
 
# Fill missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Feature Engineering: Create FamilySize
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
 
# Encode categorical variables
encoder = LabelEncoder()
df["Sex"] = encoder.fit_transform(df["Sex"])  # Male: 1, Female: 0
df["Embarked"] = encoder.fit_transform(df["Embarked"])  # C, Q, S -> 0, 1, 2
 
# Normalize numerical features
scaler = StandardScaler()
df[["Age", "Fare", "FamilySize"]] = scaler.fit_transform(df[["Age", "Fare", "FamilySize"]])
 
# Define features and target
X = df.drop("Survived", axis=1)
y = df["Survived"]
 
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Hyperparameter tuning for Random Forest
rf_params = {"n_estimators": [50, 500, 1000], "max_depth": [5, 10, None], "min_samples_split": [2, 5, 10]}
rf = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, n_jobs=-1)
rf.fit(X_train, y_train)
best_rf = rf.best_estimator_
 
# Define additional models
gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42)
xgb = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42, use_label_encoder=False, eval_metric='logloss')
 
# Train ensemble model
ensemble = VotingClassifier(estimators=[("RandomForest", best_rf), ("GradientBoosting", gb), ("XGBoost", xgb)], voting="soft")
ensemble.fit(X_train, y_train)
 
# Make predictions
y_pred = ensemble.predict(X_test)
 
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
 
# Save predictions to Excel
predictions_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
predictions_df.to_excel("Titanic_Predictions.xlsx", index=False)
 
# Visualization 1: Survival Rate by Gender
plt.figure(figsize=(6,4))
sns.barplot(x="Sex", y="Survived", data=df)
plt.xticks([0, 1], ["Female", "Male"])
plt.title("Survival Rate by Gender")
plt.show(block=False)
 
# Visualization 2: Survival Rate by Class
plt.figure(figsize=(6,4))
sns.barplot(x="Pclass", y="Survived", data=df)
plt.title("Survival Rate by Class")
plt.show(block=False)
 
# Visualization 3: Age Distribution of Survivors vs Non-Survivors
plt.figure(figsize=(8,6))
sns.histplot(df[df["Survived"] == 1]["Age"], bins=30, label="Survived", kde=True, color="green")
sns.histplot(df[df["Survived"] == 0]["Age"], bins=30, label="Not Survived", kde=True, color="red")
plt.legend()
plt.title("Age Distribution of Survivors vs Non-Survivors")
plt.show(block=False)
 
# Visualization 4: Feature Importance (Random Forest)
importances = best_rf.feature_importances_
features = X.columns
plt.figure(figsize=(8,6))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance in Survival Prediction")
plt.show(block=False)
 
# Visualization 5: Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Survived", "Survived"], yticklabels=["Not Survived", "Survived"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
