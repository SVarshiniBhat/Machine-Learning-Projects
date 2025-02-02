import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

# Load the dataset
df = pd.read_csv(r"C:\Users\Dell\Desktop\credit fraud\data\raw\creditcard.csv")

# Preprocessing: Check for missing values
print("Missing values in dataset:")
print(df.isnull().sum())

# Feature scaling: Apply StandardScaler to the features
scaler = StandardScaler()
X = df.drop(columns=['Class'])  # All features except 'Class'
y = df['Class']  # Target variable is 'Class'

# Scale the features
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

# Train a simple Random Forest Classifier
rf_model = RandomForestClassifier(
    n_estimators=100,  # Reasonable default for trees
    max_depth=20,      # Limiting depth to avoid overfitting
    random_state=42
)
rf_model.fit(X_res, y_res)

# Make predictions
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]  # Probability of fraud

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Random Forest Model Performance:")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Plot the Precision-Recall Curve
precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals, marker='.')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

# Plot the Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Saving the cleaned data (optional)
df_cleaned = df.copy()
df_cleaned.to_csv(r'C:\Users\Dell\Desktop\credit fraud\data\processed\cleaned_credit_data.csv', index=False)
print("Cleaned data saved to: C:\\Users\\Dell\\Desktop\\credit fraud\\data\\processed\\cleaned_credit_data.csv")
