import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample

# Load dataset
file_path = "heart_attack_youngsters_india_final.csv"  # Update with your actual path
df = pd.read_csv(file_path)

# Encode categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove("Blood Pressure (systolic/diastolic mmHg)")
categorical_cols.remove("Heart Attack Likelihood")

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Extract Systolic & Diastolic Blood Pressure
df[['Systolic_BP', 'Diastolic_BP']] = df["Blood Pressure (systolic/diastolic mmHg)"].str.split('/', expand=True).astype(float)
df.drop(columns=["Blood Pressure (systolic/diastolic mmHg)"], inplace=True)

# Encode target variable
df["Heart Attack Likelihood"] = df["Heart Attack Likelihood"].map({"Yes": 1, "No": 0})

# Handle class imbalance using upsampling
df_majority = df[df["Heart Attack Likelihood"] == 0]
df_minority = df[df["Heart Attack Likelihood"] == 1]

df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
df_balanced = pd.concat([df_majority, df_minority_upsampled])

# Define features and target
X = df_balanced.drop(columns=["Heart Attack Likelihood"])
y = df_balanced["Heart Attack Likelihood"]

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)

# Predictions
y_pred = best_rf.predict(X_test)

# Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Optimized Model Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)
