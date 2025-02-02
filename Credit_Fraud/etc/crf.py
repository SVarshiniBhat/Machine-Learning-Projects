import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r"C:\Users\Dell\Desktop\credit fraud\data\raw\creditcard.csv")

# Preprocessing: Check for missing values
print("Missing values in dataset:")
print(df.isnull().sum())

# Imputation for missing values (if any) - in this case there are no missing values
# But if required, you can use fillna() for missing data
# df.fillna(df.mean(), inplace=True)

# Feature scaling: Apply StandardScaler to the features
scaler = StandardScaler()
X = df.drop(columns=['Class'])  # All features except 'Class'
y = df['Class']  # Target variable is 'Class'

# Scale the features
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Handle class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

# Train the Naive Bayes model (GaussianNB for continuous features)
nb_model = GaussianNB()
nb_model.fit(X_res, y_res)

# Make predictions on the test set
y_pred = nb_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Naive Bayes Model Performance:")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Plot the distribution of Fraud vs Non-Fraud transactions after SMOTE
plt.figure(figsize=(6, 4))  # Set the figure size
sns.countplot(x=y_res, palette='Set2',hue=y_res)
plt.title("Fraud vs Non-Fraud Distribution After SMOTE")
plt.xlabel("Transaction Type")
plt.ylabel("Count")
plt.xticks([0, 1], ['Non-Fraud', 'Fraud'], rotation=0)  # Customizing x-axis labels
plt.show()

# Confusion Matrix for Naive Bayes
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Naive Bayes Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Saving the cleaned data (you can skip this if you're not modifying the dataset)
df_cleaned = df.copy()
df_cleaned.to_csv(r'C:\Users\Dell\Desktop\credit fraud\data\processed\cleaned_credit_data', index=False)
print("Cleaned data saved to: path/to/cleaned_credit_data.csv")
