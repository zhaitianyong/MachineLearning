import numpy as np
from math import sqrt

'''accuracy ratio'''


def compute_accuracy_ratio(y_test, y_predict):
    assert y_predict.shape[0] == y_test.shape[0], \
        "they both should be same shape"
    return sum(y_test == y_predict) / len(y_test)


'''MSE'''


def mean_square_error(y_test, y_predict):
    return np.sum((y_test - y_predict) ** 2) / len(y_test)


'''RMSE'''


def root_mean_square_error(y_test, y_predict):
    return sqrt(mean_square_error(y_test, y_predict))


'''平均绝对误差（mean absolute error）'''


def mean_absolute_error(y_test, y_predict):
    return np.sum(np.absolute(y_test - y_predict)) / len(y_test)


'''R square'''


def r2_score(y_test, y_predict):
    ## return 1 - np.sum((y_test - y_predict) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    return 1 - mean_square_error(y_test, y_predict) / np.var(y_test)
