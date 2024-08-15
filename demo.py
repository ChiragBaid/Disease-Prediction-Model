import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier    
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load data
data = pd.read_csv('2022/heart_2022_with_nans.csv')

# Print the first few rows of the data
print(data.head())

# Separate features and target
x = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Remove rows with missing values
x = x.dropna()
y = y[x.index]

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize the model
model = DecisionTreeClassifier()

# Convert categorical variables to numeric using one-hot encoding
x_train = pd.get_dummies(x_train)
x_test = pd.get_dummies(x_test)

# Align train and test sets to ensure they have the same columns
x_train, x_test = x_train.align(x_test, join='left', axis=1, fill_value=0)

# Train the model
model.fit(x_train, y_train)

# Predict on the test set
y_pred = model.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(model, 'disease_prediction_model.pkl')
