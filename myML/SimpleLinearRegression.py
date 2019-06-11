import numpy as np
from .metrics import r2_score

class SimpleLinearRegression:
    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        assert x_train.ndim == 1, \
            "can only one dim"
        assert len(x_train) == len(y_train), \
            "the size of x_train must be equal the size of y_train"
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = 0.0
        d = 0.0
        # for x, y in zip(x_train, y_train):
        #     num += (x - x_mean) * (y - y_mean)
        #     d += (x - x_mean) ** 2
        num = (x_train - x_mean).dot(y_train - y_mean)
        d = (x_train - x_mean).dot(x_train - x_mean)

        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean

        return self.a_, self.b_

    def predict(self, x_predict):
        assert x_predict.ndim == 1, \
            "can only one dim"
        return self.a_ * x_predict + self.b_

    def score(self, x_test, y_test):
        y_predict = self.predict(x_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "SimpleLinearRegression"
