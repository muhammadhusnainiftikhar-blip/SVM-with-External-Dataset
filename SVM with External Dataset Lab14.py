
# Lab 14: SVM Using Self-Created / Kaggle Dataset

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

from google.colab import files

# Step 2: Upload Dataset
uploaded = files.upload()

# Step 3: Load Dataset
df = pd.read_csv(list(uploaded.keys())[0])

# Rename Columns
# Dataset format:
# Date , Price

df.columns = ["Date", "Price"]

print("Dataset Preview:")
print(df.head())

# Step 4: Handle Missing Values
df = df.dropna()

# Step 5: Convert Date into Numeric Format
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Remove invalid rows
df = df.dropna()

# Convert date into number of days
df["Date_num"] = (
    df["Date"] - df["Date"].min()
).dt.days

# ---------------------------------------------------
# Create Classification Labels
# If Price > Average Price = 1
# Else = 0
# ---------------------------------------------------

average_price = df["Price"].mean()

df["Target"] = (
    df["Price"] > average_price
).astype(int)

print("\nProcessed Dataset:")
print(df.head())

# Step 6: Split Input and Output
X = df[["Date_num"]].values
y = df["Target"].values

# Step 7: Split Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Step 8: Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 9: Create SVM Model
model = SVC(kernel='linear')

# Step 10: Train Model
model.fit(X_train, y_train)

# Step 11: Predict Results
y_pred = model.predict(X_test)

print("\nPredicted Values:")
print(y_pred)

# Step 12: Check Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy)

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ---------------------------------------------------
# Visualization
# ---------------------------------------------------

plt.figure(figsize=(8,6))

plt.scatter(
    X,
    y,
    color='blue',
    label='Dataset'
)

plt.title("SVM Classification using CSV Dataset")
plt.xlabel("Days")
plt.ylabel("Class")

plt.grid(True)
plt.legend()

plt.show()

print("\nSVM Model Executed Successfully!")