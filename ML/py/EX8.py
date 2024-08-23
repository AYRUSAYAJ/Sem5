import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

lda = LDA(n_components=2) 
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue']
labels = iris.target_names

for i, color, label in zip(np.unique(y_train), colors, labels):
    plt.scatter(X_train_lda[y_train == i, 0], X_train_lda[y_train == i, 1], c=color, label=label, edgecolor='k')

plt.xlabel('LDA Component 1')
plt.ylabel('LDA Component 2')
plt.title('LDA on Iris Dataset')
plt.legend(loc='best')
plt.show()

y_pred = lda.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
