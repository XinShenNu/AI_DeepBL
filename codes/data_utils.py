import numpy as np
import pandas as pd


def mean_absolute_percentage_error(y_true, y_pred):
    temp = 0
    for i in range(0, y_true.size):
        if y_true[i] != 0:
            temp += np.abs((y_true[i] - y_pred[i]) / y_true[i]) * 100
        else:
            if y_pred[i] == y_true[i]:
                temp += 0
            else:
                temp += np.abs((y_true[i] - y_pred[i]) / y_pred[i]) * 100
    return temp * 1.0 / y_true.size


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    temp = 0
    for i in range(0, y_true.size):
        if np.abs(y_true[i]) + np.abs(y_pred[i]) != 0:
            temp += 100 * np.abs(y_true[i] - y_pred[i]) / ((np.abs(y_true[i]) + np.abs(y_pred[i])) / 2)
    return temp * 1.0 / y_true.size
