import pandas as pd
import math
import numpy as np

def value(data):
    def f(i):
        return np.c_[data.df.loc[i, data.features_ohe]]
    return f

def matrixmultiply(matrix):
    def f(vector):
        return matrix * vector
    return f

def sigmoid(x):
    return 1 / (1 + math.exp(x))

def sigmoid_v(x):
    return np.vectorize(sigmoid)(x)