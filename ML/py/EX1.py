import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold ,StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("../diabetes.csv")

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)
accuracy_before_cv = accuracy_score(y_test, y_pred)
print(f'Accuracy before cross-validation: {accuracy_before_cv}')

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(dt, X, y, cv=kf, scoring='accuracy')
accuracy_after_cv = cv_scores.mean()
print(f'Accuracy after k-fold cross-validation: {accuracy_after_cv}')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(dt, X, y, cv=skf, scoring='accuracy')
accuracy_after_cv = cv_scores.mean()
print(f'Accuracy after stratified cross-validation: {accuracy_after_cv}')