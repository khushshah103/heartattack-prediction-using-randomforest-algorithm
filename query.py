import pandas as pd

# Load the dataset with enhanced error handling
file_path = 'heart_attack_youngsters_india_final.csv'

try:
    df = pd.read_csv(file_path, encoding_errors='ignore')
    print("File loaded successfully!")
except Exception as e:
    print(f"Error loading file: {e}")

# Display column names for debugging
print("Columns in dataset:", df.columns.tolist())

# Ensure correct column names
if 'Triglyceride Levels' not in df.columns:
    raise KeyError("The required column 'Triglyceride Levels' is missing from the dataset.")

# Calculate the total number of people with Triglyceride Levels > 200
triglyceride_above_200 = (df['Triglyceride Levels'] > 200).sum()
print(f"Total number of people with Triglyceride Levels > 200: {triglyceride_above_200}")
