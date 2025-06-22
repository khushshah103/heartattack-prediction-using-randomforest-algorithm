import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, chi2
from collections import Counter
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
file_path = 'heart_attack_youngsters_india_final.csv'
df = pd.read_csv(file_path)

# Convert 'Heart Attack Likelihood' to binary (0 and 1)
# Adjust mapping based on actual values, e.g., 'Low'->0, 'High'->1 or 'No'->0, 'Yes'->1
df['Heart Attack Likelihood'] = df['Heart Attack Likelihood'].map({'No': 0, 'Yes': 1})

# Check class distribution
print("Class distribution:", Counter(df['Heart Attack Likelihood']))

# Separate features and target
X = df.drop('Heart Attack Likelihood', axis=1)
y = df['Heart Attack Likelihood']

# One-hot encode categorical features
X = pd.get_dummies(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature selection (top 100 features)
selector = SelectKBest(chi2, k=100)
X_train_reduced = selector.fit_transform(X_train, y_train)
X_test_reduced = selector.transform(X_test)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_reduced)
X_test_scaled = scaler.transform(X_test_reduced)

# Evaluation function
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred, zero_division=1) * 100
    recall = recall_score(y_test, y_pred, zero_division=1) * 100
    f1 = f1_score(y_test, y_pred, zero_division=1) * 100

    print(f"{model.__class__.__name__} Results:")
    print("Accuracy: {:.2f}%".format(accuracy))
    print("Precision: {:.2f}%".format(precision))
    print("Recall: {:.2f}%".format(recall))
    print("F1 Score: {:.2f}%".format(f1))
    print("---------------------")

# Logistic Regression
logistic_model = LogisticRegression(class_weight='balanced', max_iter=1000)
evaluate_model(logistic_model, X_train_scaled, X_test_scaled, y_train, y_test)

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
evaluate_model(dt_model, X_train_scaled, X_test_scaled, y_train, y_test)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
evaluate_model(rf_model, X_train_scaled, X_test_scaled, y_train, y_test)

# XGBoost
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
evaluate_model(xgb_model, X_train_scaled, X_test_scaled, y_train, y_test)

# k-Nearest Neighbors
knn_model = KNeighborsClassifier(n_neighbors=5)
evaluate_model(knn_model, X_train_scaled, X_test_scaled, y_train, y_test)

# Deep Learning Model
dl_model = keras.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

dl_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
dl_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate DL model
y_pred_dl = (dl_model.predict(X_test_scaled) > 0.5).astype("int32")

accuracy = accuracy_score(y_test, y_pred_dl) * 100
precision = precision_score(y_test, y_pred_dl, zero_division=1) * 100
recall = recall_score(y_test, y_pred_dl, zero_division=1) * 100
f1 = f1_score(y_test, y_pred_dl, zero_division=1) * 100

print("Deep Learning Model Results:")
print("Accuracy: {:.2f}%".format(accuracy))
print("Precision: {:.2f}%".format(precision))
print("Recall: {:.2f}%".format(recall))
print("F1 Score: {:.2f}%".format(f1))
