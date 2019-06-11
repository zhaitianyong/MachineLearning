import numpy as np
from math import sqrt
from collections import Counter
from metrics import compute_accuracy_ratio

class kNNClassifier:
    def __init__(self, k):
        self._k = k
        self._x_train = None
        self._y_train = None

    def fit(self, x_train, y_train):
        self._x_train = x_train
        self._y_train = y_train

    def predict(self, X_test):
        y_predict = [self._predict(x) for x in X_test]
        return np.array(y_predict)

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return compute_accuracy_ratio(y_test, y_predict)

    def _predict(self, x):
        distances = [sqrt(np.sum((item - x) ** 2)) for item in self._x_train]
        nearest = np.argsort(distances)
        topK_y = [self._y_train[i] for i in nearest[:self._k]]
        votes = Counter(topK_y)
        return votes.most_common(1)[0][0]

    def __repr__(self):
        print("kNN(k=%d)", self._k)
