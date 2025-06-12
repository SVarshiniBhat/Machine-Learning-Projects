import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# --- Training the model ---
def train_model():
    # Load the dataset
    df = pd.read_csv(r"C:\Users\Dell\OneDrive\Documents\CodSoft\Credit_Fraud\data\raw\creditcard.csv")

    # Preprocessing: Check for missing values
    print("Checking for missing values in the dataset...")
    if df.isnull().sum().sum() == 0:
        print("No missing values found.\n")
    else:
        print("Missing values detected:\n", df.isnull().sum())

    # Feature scaling: Apply StandardScaler to the features
    scaler = StandardScaler()
    X = df.drop(columns=['Class'])  # All features except 'Class'
    y = df['Class']  # Target variable

    # Scale the features
    X_scaled = scaler.fit_transform(X)

    # Apply SMOTE to balance the dataset
    print("Applying SMOTE to balance the dataset...")
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier
    print("Training the Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=100,   # Number of trees
        max_depth=20,       # Max depth of each tree
        class_weight='balanced',
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    # Make predictions with probabilities
    y_prob = rf_model.predict_proba(X_test)[:, 1]  # Probability of fraud

    # Adjust the threshold for classification
    threshold = 0.1  # Adjust this to balance precision and recall
    y_pred = (y_prob >= threshold).astype(int)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\nRandom Forest Model Performance:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}\n")

    # Save the trained model and scaler
    try:
        model_path = r'C:\Users\Dell\Desktop\credit fraud\models\random_forest_model.pkl'
        scaler_path = r'C:\Users\Dell\Desktop\credit fraud\models\scaler.pkl'
        joblib.dump(rf_model, model_path)
        joblib.dump(scaler, scaler_path)
        print(f"Model saved successfully to {model_path}")
        print(f"Scaler saved successfully to {scaler_path}\n")
    except Exception as e:
        print(f"Error saving model and scaler: {e}")

    # Plotting Confusion Matrix (Normalized)
    plt.figure(figsize=(8, 6))
    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    sns.heatmap(conf_matrix_normalized, annot=True, fmt=".2f", cmap="Blues", cbar=True,
                annot_kws={"size": 14}, linewidths=0.5)
    plt.title("Confusion Matrix (Normalized)", fontsize=16)
    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("Actual", fontsize=14)
    plt.xticks([0.5, 1.5], labels=["Non-Fraud", "Fraud"], fontsize=12)
    plt.yticks([0.5, 1.5], labels=["Non-Fraud", "Fraud"], fontsize=12)

    # Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_prob)
    plt.plot(recall_vals, precision_vals, marker='.', label=f'Threshold = {threshold}')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.legend()

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

    # Save the cleaned data
    df_cleaned = df.copy()
    df_cleaned.to_csv(r'C:\Users\Dell\OneDrive\Documents\CodSoft\Credit_Fraud\data\processed\cleaned_credit_data.csv', index=False)
    print("Cleaned data saved to: cleaned_credit_data.csv")


# --- Testing with new data ---
def test_model_with_new_data(new_data_path):
    # Load the trained model and scaler
    model = joblib.load(r'C:\Users\Dell\OneDrive\Documents\CodSoft\Credit_Fraud\models\random_forest_model.pkl')
    scaler = joblib.load(r'C:\Users\Dell\OneDrive\Documents\CodSoft\Credit_Fraud\models\scaler.pkl')

    # Load new data to test
    new_data = pd.read_csv(new_data_path)

    # Scale the features
    new_data_scaled = scaler.transform(new_data.drop(columns=['Class']))

    # Make predictions
    probabilities = model.predict_proba(new_data_scaled)[:, 1]  # Probability of fraud
    predictions = (probabilities >= 0.1).astype(int)  # Using threshold of 0.1

    # Save the predictions to a CSV
    new_data['Predicted_Class'] = predictions
    new_data['Probability'] = probabilities
    output_path = r'C:\Users\Dell\OneDrive\Documents\CodSoft\Credit_Fraud\data\processed\predictions.csv'
    new_data.to_csv(output_path, index=False)

    # Count fraud and non-fraud predictions
    fraud_count = (predictions == 1).sum()
    non_fraud_count = (predictions == 0).sum()

    print(f"Predictions saved to: {output_path}")
    print(f"Number of predicted fraud transactions: {fraud_count}")
    print(f"Number of predicted non-fraud transactions: {non_fraud_count}")


# --- Main ---
if __name__ == "__main__":
    train_model()

    # Test with new data (update the path to the new data)
    new_data_path = r'C:\Users\Dell\OneDrive\Documents\CodSoft\Credit_Fraud\data\raw\generated_credit_data.csv'
    test_model_with_new_data(new_data_path)

    # Show all plots
    plt.show()
