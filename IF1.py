import pandas as pd
import math
import numpy as np

def colvec(data):
    def f(i):
        return np.c_[data.df.loc[i, data.features_ohe]]
    return f

def matrixmultiply(matrix):
    def f(vector):
        return matrix @ vector
    return f

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_v(x):
    return np.vectorize(sigmoid)(x)

def rand_w(n, m):
    return (.01 - (-.01)) * np.random.random_sample((n, m)) - .01

def predict_value(data):
    w = rand_w(1, data.df.shape[1] - 1)
    c = colvec(data)
    def f(i):
        return (w @ c(i))[0, 0]
    return f

def error(data):
    pred_func = predict_value(data)
    actual_func = lambda i: data.df.at[i, "Target"]
    def f(i):
        predicted = pred_func(i)
        actual = actual_func(i)
        return .5 * math.pow(actual - predicted, 2)
    return f
