import pandas as pd
import math
import numpy as np
import functools

def vec(data):
    def f(i):
        return np.concatenate(([1], data.df.loc[i, data.features_ohe]))
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
    w_init = rand_w(1, data.df.shape[1])
    vec_func = vec(data)
    actual_func = lambda i: data.df.at[i, "Target"]
    def f1(eta, eps):
        def f2(i):
            r = actual_func(i)
            x = vec_func(i)
            def f3(wi = w_init):
                def f3_rec(w):
                    y = (w @ x.reshape(-1, 1))[0, 0]
                    d = r - y
                    if abs(d) < eps: return y
                    dw = np.vectorize(lambda xj: eta * d * xj)(x)
                    return f3_rec(w + dw)
                return f3_rec(wi)
            return f3
        return f2
    return f1




