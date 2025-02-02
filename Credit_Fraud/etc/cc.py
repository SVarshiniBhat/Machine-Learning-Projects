import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
import seaborn as sns
import matplotlib.pyplot as plt

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

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(
    n_estimators=100,   # Number of trees
    max_depth=20,       # Max depth of each tree
    class_weight='balanced',  # Handle class imbalance directly
    random_state=42
)
rf_model.fit(X_train, y_train)

# Make predictions with probabilities
y_prob = rf_model.predict_proba(X_test)[:, 1]  # Probability of fraud

# Adjust the threshold for classification
threshold = 0.2  # Adjust this to balance precision and recall
y_pred = (y_prob >= threshold).astype(int)

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


# Plot the Confusion Matrix with normalization
plt.figure(figsize=(8, 6))
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]  # Normalize by row

sns.heatmap(conf_matrix_normalized, annot=True, fmt=".2f", cmap="Blues", cbar=True, 
            annot_kws={"size": 16}, linewidths=0.5)
plt.title("Random Forest Confusion Matrix (Normalized)", fontsize=16)
plt.xlabel("Predicted", fontsize=14)
plt.ylabel("Actual", fontsize=14)
plt.xticks(ticks=[0.5, 1.5], labels=["Non-Fraud", "Fraud"], fontsize=12)
plt.yticks(ticks=[0.5, 1.5], labels=["Non-Fraud", "Fraud"], fontsize=12)


# Precision-Recall Curve
plt.figure(figsize=(8, 6))
precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_prob)
plt.plot(recall_vals, precision_vals, marker='.', label=f'Threshold = {threshold}')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(True)
plt.legend()
# plt.show()

# ROC Curve
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)

# plt.ion()
plt.show()


# Saving the cleaned data (optional)
df_cleaned = df.copy()
df_cleaned.to_csv(r'C:\Users\Dell\Desktop\credit fraud\data\processed\cleaned_credit_data.csv', index=False)
print("Cleaned data saved to: C:\\Users\\Dell\\Desktop\\credit fraud\\data\\processed\\cleaned_credit_data.csv")
