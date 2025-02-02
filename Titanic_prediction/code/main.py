import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
 
# Load dataset
df = pd.read_csv(r"C:\Users\Dell\Desktop\titanic_prediction\data\raw\Titanic-Dataset.csv")
 
# Drop unnecessary columns
df.drop(["PassengerId", "Ticket", "Cabin"], axis=1, inplace=True)
 
# Fill missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Fare"] = df["Fare"].fillna(df["Fare"].median())
 

# Feature Engineering
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1  # Total number of family members
df["FarePerPerson"] = df["Fare"] / df["FamilySize"]
 
# Extract Title from Name
df["Title"] = df["Name"].apply(lambda x: x.split(",")[1].split(".")[0].strip())
df["Title"] = df["Title"].replace(["Mlle", "Ms", "Mme"], "Miss")
df["Title"] = df["Title"].replace(["Capt", "Col", "Major", "Dr", "Rev", "Jonkheer", "Don", "Sir", "Lady", "Countess"], "Rare")
df.drop(["Name"], axis=1, inplace=True)
 
# Encode categorical variables
encoder = LabelEncoder()
df["Sex"] = encoder.fit_transform(df["Sex"])  # Male: 1, Female: 0
df["Embarked"] = encoder.fit_transform(df["Embarked"])  # C, Q, S -> 0, 1, 2
df["Title"] = encoder.fit_transform(df["Title"])  # Encodes Titles
 
# Normalize numerical features
scaler = StandardScaler()
df[[ "Fare", "FamilySize", "FarePerPerson"]] = scaler.fit_transform(df[[ "Fare", "FamilySize", "FarePerPerson"]])

# Reverse the scaling (inverse transform)
original_values = scaler.inverse_transform(df[[ 'Fare', 'FamilySize', 'FarePerPerson']])
df[['Fare', 'FamilySize', 'FarePerPerson']] = original_values

# Define features and target
X = df.drop("Survived", axis=1)
y = df["Survived"]
 
# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
 
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
 
# Train base models
rf = RandomForestClassifier(n_estimators=1000, max_depth=10, min_samples_split=5, random_state=42)
gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42)
xgb = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42, eval_metric='logloss')
 
# Stacking Classifier (Advanced Ensemble)
stacking_model = StackingClassifier(
    estimators=[("RandomForest", rf), ("GradientBoosting", gb), ("XGBoost", xgb)],
    final_estimator=LogisticRegression(max_iter=1000),
    passthrough=True
)
 
# Train stacked model
stacking_model.fit(X_train, y_train)
 
# Make predictions
y_pred = stacking_model.predict(X_test)
 
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save cleaned dataset to the processed directory
processed_path = r"C:\Users\Dell\Desktop\titanic_prediction\data\processed\cleaned_titanic_data.csv"
df.to_csv(processed_path, index=False)
print(f"✅ Cleaned dataset saved to {processed_path}")

print("============================================================")
 
# Save predictions to Excel
predictions_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
predictions_df.to_excel("Titanic_Predictions_Improved.xlsx", index=False)
print("✅ Predictions saved to Titanic_Predictions_Improved.xlsx")
 

# Fit RandomForestClassifier for feature importance calculation
rf.fit(X_train, y_train)

# Visualization 1: Feature Importance (Random Forest)
importances = rf.feature_importances_
features = X.columns
plt.figure(figsize=(8, 6))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance in Survival Prediction")
plt.show(block=False)
 
# Visualization 2: Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Survived", "Survived"], yticklabels=["Not Survived", "Survived"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
 

























# # Visualization 1: Survival Rate by Gender
# plt.figure(figsize=(6,4))
# sns.barplot(x="Sex", y="Survived", data=df)
# plt.xticks([0, 1], ["Female", "Male"])
# plt.title("Survival Rate by Gender")
# plt.show()
 
# # Visualization 2: Survival Rate by Class
# plt.figure(figsize=(6,4))
# sns.barplot(x="Pclass", y="Survived", data=df)
# plt.title("Survival Rate by Class")
# plt.show()
 
# # Visualization 3: Age Distribution of Survivors vs Non-Survivors
# plt.figure(figsize=(8,6))
# sns.histplot(df[df["Survived"] == 1]["Age"], bins=30, label="Survived", kde=True, color="green")
# sns.histplot(df[df["Survived"] == 0]["Age"], bins=30, label="Not Survived", kde=True, color="red")
# plt.legend()
# plt.title("Age Distribution of Survivors vs Non-Survivors")
# plt.show()
 