# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1: Import the required libraries and load the dataset containing employee details such as Experience, Education, Department, and Salary.

2: Preprocess the data by encoding categorical variables and scaling numerical features using ColumnTransformer.

3: Build and train the DecisionTreeRegressor model using a pipeline, and tune hyperparameters with GridSearchCV for best performance.

4: Evaluate the model using performance metrics (MAE, MSE, R²), make salary predictions for new data, and visualize the decision tree and feature importance.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Sanjana K L  
RegisterNumber:  212224230241
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = {
    'Experience': [1, 3, 5, 7, 9, 11, 13, 15, 17, 20],
    'Education': ['High School', 'Bachelors', 'Masters', 'PhD', 'Bachelors', 
                  'Masters', 'PhD', 'Bachelors', 'Masters', 'PhD'],
    'Department': ['HR', 'Finance', 'IT', 'IT', 'HR', 
                   'Finance', 'Finance', 'IT', 'HR', 'Finance'],
    'Salary': [30000, 40000, 55000, 75000, 85000, 95000, 110000, 125000, 140000, 160000]
}
df = pd.DataFrame(data)
print("\nSample Data:\n", df.head())

X = df[['Experience', 'Education', 'Department']]
y = df['Salary']

numeric_features = ['Experience']
categorical_features = ['Education', 'Department']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor(random_state=42))
])

param_grid = {
    'regressor__max_depth': [2, 3, 4, 5, 6],
    'regressor__min_samples_split': [2, 4, 6],
    'regressor__min_samples_leaf': [1, 2, 3]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X, y)

print("\nBest Parameters:", grid_search.best_params_)
print("Best R² Score from CV:", grid_search.best_score_)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print("\nEvaluation Metrics:")
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

cv_scores = cross_val_score(best_model, X, y, cv=5)
print("\nCross-validation Scores:", cv_scores)
print("Average CV Score:", np.mean(cv_scores))

new_employee = pd.DataFrame({
    'Experience': [10],
    'Education': ['Masters'],
    'Department': ['IT']
})
pred_salary = best_model.predict(new_employee)
print(f"\nPredicted Salary for new employee: ₹{pred_salary[0]:.2f}")

final_tree = best_model.named_steps['regressor']
plt.figure(figsize=(12, 8))
plot_tree(final_tree, filled=True, feature_names=best_model[:-1].get_feature_names_out(), rounded=True)
plt.title("Decision Tree Regressor for Salary Prediction")
plt.show()

importances = final_tree.feature_importances_
features = best_model[:-1].get_feature_names_out()
plt.figure(figsize=(8,5))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance in Salary Prediction")
plt.show()

```
## Output:
<img width="718" height="222" alt="Screenshot 2025-10-06 181932" src="https://github.com/user-attachments/assets/9b321265-f3dc-472e-8fab-d1bec3bf752a" />



<img width="1562" height="91" alt="Screenshot 2025-10-06 181949" src="https://github.com/user-attachments/assets/a62c1936-9e48-4f00-8ac9-90cf9d995557" />


<img width="486" height="163" alt="Screenshot 2025-10-06 182000" src="https://github.com/user-attachments/assets/178d08ec-2aba-40cb-bb12-3b3744045bd8" />



<img width="1348" height="161" alt="Screenshot 2025-10-06 182016" src="https://github.com/user-attachments/assets/9b1bb660-50bb-42ae-b547-8a4ac985a70f" />



<img width="1245" height="677" alt="Screenshot 2025-10-06 182048" src="https://github.com/user-attachments/assets/1992f7ee-f699-4520-9bc7-d083c11cb6ef" />



<img width="1079" height="539" alt="Screenshot 2025-10-06 182100" src="https://github.com/user-attachments/assets/49a38e3c-7bb1-4524-a323-a5525cab289a" />



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
