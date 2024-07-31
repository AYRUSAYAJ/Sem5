import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('titanic.csv')
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])
df['Embarked'] = label_encoder.fit_transform(df['Embarked'])
X = df.drop(columns=['Survived'])
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
bagging_knn = BaggingClassifier(estimator=knn, n_estimators=10, random_state=42)
bagging_knn.fit(X_train, y_train)
y_pred_knn = bagging_knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f'Bagging with KNN Accuracy: {accuracy_knn:.2f}')

randomforest = RandomForestClassifier(kernel='linear')
bagging_randomforest = BaggingClassifier(estimator=randomforest, n_estimators=10, random_state=42)
bagging_randomforest.fit(X_train, y_train)
y_pred_randomforest = bagging_randomforest.predict(X_test)
accuracy_randomforest = accuracy_score(y_test, y_pred_randomforest)
print(f'Bagging with randomforest Accuracy: {accuracy_randomforest:.2f}')

n_estimators = 60

estimators = []

for _ in range(n_estimators):
    X_resampled, y_resampled = resample(X_train, y_train, random_state=42)
    estimator = KNeighborsClassifier(n_neighbors=3)
    estimator.fit(X_resampled, y_resampled)
    estimators.append(estimator)

predictions = np.zeros((X_test.shape[0], n_estimators))

for i, estimator in enumerate(estimators):
    predictions[:, i] = estimator.predict(X_test)

final_predictions = (np.sum(predictions, axis=1) >= (n_estimators / 2)).astype(int)

final_predictions[:10]