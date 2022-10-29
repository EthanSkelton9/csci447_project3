import pandas as pd
import math
import numpy as np
import functools
import random

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

def mean_squared_error(predicted, actual):
    p_vec = predicted.to_numpy()
    a_vec = actual.to_numpy()
    diff_vec = p_vec - a_vec
    return math.pow(np.linalg.norm(diff_vec), 2) / len(predicted)

def predict_value(data):
    w_init = rand_w(1, data.df.shape[1])
    vec_func = vec(data)
    actual_func = lambda i: data.df.at[i, "Target"]
    def f1(eta, eps):
        def f2(i):
            x = vec_func(i)
            r = actual_func(i)
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

def stochastic_online_gd(data, n):
    w_init = rand_w(1, data.df.shape[1])
    vec_func = vec(data)
    actual_func = lambda i: data.df.at[i, "Target"]
    sub_index = random.sample(list(data.df.index), k=n)
    df = pd.DataFrame(data.df.filter(items=sub_index, axis=0).to_dict())
    r = df["Target"]
    y = pd.Series(n * [None], index=sub_index)
    def f1(eta, eps):
        def f1_acc(count, index, w):
            if count == n:
                if mean_squared_error(y, r) < eps: return y
                return f1_acc(0, random.sample(sub_index, n), w)










