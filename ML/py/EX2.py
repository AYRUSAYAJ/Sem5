import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("../diabetes.csv")

x = df.drop('Outcome', axis=1)
y = df['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)

accuracy_before_cv = accuracy_score(y_test, y_pred)
print(f'Accuracy before cross-validation: {accuracy_before_cv}')

n_splits = 5

indices = np.arange(len(x))

fold_indices = np.array_split(indices, n_splits)

cv_scores = []
for i in range(n_splits):

    test_indices = fold_indices[i]
    train_indices = np.concatenate([fold_indices[j] for j in range(n_splits) if j != i])
    
    x_train, x_test = x.iloc[train_indices], x.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    
    dt.fit(x_train, y_train)
    
    y_pred = dt.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores.append(accuracy)

accuracy_after_cv = np.mean(cv_scores)
print(f'Accuracy after manual cross-validation: {accuracy_after_cv}')