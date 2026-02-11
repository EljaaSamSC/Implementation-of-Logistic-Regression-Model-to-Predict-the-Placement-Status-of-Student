# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("Placement_Data.csv")  
print("Dataset Preview:")
print(data.head())
data = data.drop(["sl_no", "salary"], axis=1)
data["status"] = data["status"].map({"Placed": 1, "Not Placed": 0})
X = data.drop("status", axis=1)
y = data["status"]
X = pd.get_dummies(X, drop_first=True)
print("\nAfter Encoding:")
print(X.head())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Placement Prediction")
plt.show()
```

## Output:
<img width="990" height="630" alt="image" src="https://github.com/user-attachments/assets/4c0ab34e-aa42-4b43-b94d-181af8c79db3" />

<img width="876" height="432" alt="image" src="https://github.com/user-attachments/assets/823132a0-9ac9-41c3-9b40-f972b17dd8ee" />

<img width="837" height="599" alt="image" src="https://github.com/user-attachments/assets/17c3efdc-b108-4084-87a7-e6fdd1d4095a" />

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
