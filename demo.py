import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier   
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
data = pd.read_csv('2020/heart_2022_no_nans.csv')

# Display the first few rows of the dataset
print(data.head())

# Split the dataset into features (X) and target variable (y)
x = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Handle any missing values by filling with zero
if x_train.isnull().values.any() or x_test.isnull().values.any():
    print("NaN values are still present in the dataset after imputation.")
    x_train = x_train.fillna(0)
    x_test = x_test.fillna(0)

# Convert categorical variables to dummy/indicator variables
x_train = pd.get_dummies(x_train)
x_test = pd.get_dummies(x_test)

# Align the train and test sets to ensure they have the same columns
x_train, x_test = x_train.align(x_test, join='left', axis=1, fill_value=0)

# Initialize the RandomForestClassifier model
model = RandomForestClassifier(random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(x_train, y_train)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Fit the best model to the training data
best_model.fit(x_train, y_train)

# Perform cross-validation
cv_scores = cross_val_score(best_model, x_train, y_train, cv=5)
print(f'Cross-validation accuracy: {np.mean(cv_scores):.2f}')

# Make predictions on the test data
y_pred = best_model.predict(x_test)

# Calculate accuracy and display the classification report
accuracy = accuracy_score(y_test, y_pred)
print(f'Final Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Save the trained model to a file
joblib.dump(best_model, 'disease_prediction_model.pkl')
