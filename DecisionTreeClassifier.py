import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, chi2
from collections import Counter

# Load the dataset
file_path = 'heart_attack_youngsters_india_final.csv'
df = pd.read_csv(file_path)

# Check class distribution
print("Class distribution:", Counter(df['Heart Attack Likelihood']))

# Encode the target variable ('Heart Attack Likelihood')
label_encoder = LabelEncoder()
df['Heart Attack Likelihood'] = label_encoder.fit_transform(df['Heart Attack Likelihood'])

# Separate features (X) and target (y)
X = df.drop('Heart Attack Likelihood', axis=1)
y = df['Heart Attack Likelihood']

# Encode categorical variables using one-hot encoding
X = pd.get_dummies(X)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature selection: Select top 100 features
selector = SelectKBest(chi2, k=100)
X_train_reduced = selector.fit_transform(X_train, y_train)
X_test_reduced = selector.transform(X_test)

# Train the Decision Tree model with class balancing
model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
model.fit(X_train_reduced, y_train)

# Make predictions
y_pred = model.predict(X_test_reduced)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1)
recall = recall_score(y_test, y_pred, zero_division=1)
f1 = f1_score(y_test, y_pred, zero_division=1)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
