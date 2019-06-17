# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
# import numpy as np

iris_dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'],random_state=0)

# iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker ='o', hist_kwds={'bins': 20}, s=60, alpha=.8)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# X_new = np.array([[5, 2.9, 1, 0.2]])
# prediction = knn.predict(X_new)

score = knn.score(X_test, y_test)
