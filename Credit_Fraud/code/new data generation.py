import pandas as pd
import numpy as np

# Set the seed for reproducibility
np.random.seed(42)

# Number of rows in the dataset
n_rows = 1000

# Generate 'Time' column with random values
time = np.random.uniform(0, 500, n_rows)

# Generate 'V1' to 'V28' columns with random values
v_columns = [f'V{i}' for i in range(1, 29)]
v_data = np.random.randn(n_rows, 28)

# Generate 'Amount' column with random values as integers or floats (you can choose based on your needs)
amount = np.random.uniform(10, 2000, n_rows)

# Generate 'Class' column (0 for non-fraud, 1 for fraud)
class_labels = np.random.choice([0, 1], size=n_rows, p=[0.70, 0.30])

# Combine all the data into a DataFrame
data = np.column_stack([time, v_data, amount, class_labels])
df = pd.DataFrame(data, columns=['Time'] + v_columns + ['Amount', 'Class'])

# Save the DataFrame as a CSV file
output_file_path = r'C:\Users\Dell\Desktop\credit fraud\data\raw\generated_credit_data.csv'
df.to_csv(output_file_path, index=False)

print(f"Generated data saved as 'generated_credit_data.csv' at {output_file_path}")
