import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV
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

# Handle class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

# Train the Naive Bayes model (GaussianNB for continuous features)
nb_model = GaussianNB()

# Optional: Hyperparameter tuning (although not as common with Naive Bayes)
param_grid = {
    'var_smoothing': np.logspace(0,-9, num=100)  # Hyperparameter to adjust smoothing
}
grid_search = GridSearchCV(estimator=nb_model, param_grid=param_grid, cv=5, verbose=1, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_res, y_res)
best_model = grid_search.best_estimator_

# Train the best model found through GridSearchCV
best_model.fit(X_res, y_res)

# Make predictions on the test set and predict probabilities for threshold adjustments
y_prob = best_model.predict_proba(X_test)[:, 1]  # Probability of class 1 (fraud)

# Calculate precision-recall curve
precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_prob)

# Find the threshold with the best balance between precision and recall
# We will calculate the F1 score at each threshold to find the best threshold
f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals)
best_threshold = thresholds[np.argmax(f1_scores)]

print(f"Best threshold found: {best_threshold}")

# Adjust predictions based on the best threshold
y_pred_adjusted = (y_prob >= best_threshold).astype(int)

# Evaluate the model performance with the adjusted threshold
accuracy = accuracy_score(y_test, y_pred_adjusted)
precision = precision_score(y_test, y_pred_adjusted)
recall = recall_score(y_test, y_pred_adjusted)
f1 = f1_score(y_test, y_pred_adjusted)

print("Naive Bayes Model Performance with Best Threshold:")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Plot the Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals, marker='.')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

# Plot the distribution of Fraud vs Non-Fraud transactions after SMOTE
plt.figure(figsize=(6, 4))  # Set the figure size
sns.countplot(x=y_res, palette='Set2',hue=y_res)
plt.title("Fraud vs Non-Fraud Distribution After SMOTE")
plt.xlabel("Transaction Type")
plt.ylabel("Count")
plt.xticks([0, 1], ['Non-Fraud', 'Fraud'], rotation=0)  # Customizing x-axis labels
plt.show()

# Confusion Matrix for Naive Bayes with the best threshold
conf_matrix = confusion_matrix(y_test, y_pred_adjusted)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Naive Bayes Confusion Matrix (Best Threshold)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Saving the cleaned data (optional)
df_cleaned = df.copy()
df_cleaned.to_csv(r'C:\Users\Dell\Desktop\credit fraud\data\processed\cleaned_credit_data.csv', index=False)
print("Cleaned data saved to: path/to/cleaned_credit_data.csv")
