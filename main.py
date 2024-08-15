import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier    
from sklearn.metrics import accuracy_score, classification_report
import joblib

data = pd.read_csv('2022\heart_2022_with_nans.csv')

print(data.head())

x = data.iloc[:, :-1]
y = data.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()

if x_train.isnull().values.any() or x_test.isnull().values.any():
    print("NaN values are still present in the dataset after imputation.")
    x_train = x_train.fillna(0)
    x_test = x_test.fillna(0)

x_train = pd.get_dummies(x_train)
x_test = pd.get_dummies(x_test)

x_train, x_test = x_train.align(x_test, join='left', axis=1, fill_value=0)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

joblib.dump(model, 'disease_prediction_model.pkl')