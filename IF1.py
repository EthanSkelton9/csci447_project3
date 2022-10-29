import pandas as pd
import math
import numpy as np
import functools
import random

'''
@param data: the data set we are using
@return function that takes a row of the data and returns its numpy vector
'''
def vec(data):
    def f(i):
        return np.concatenate(([1], data.df.loc[i, data.features_ohe]))
    return f

'''
@param matrix: the matrix used to multiply
@return function that transforms vectors into vectors by multiplying by the given matrix
'''
def matrixmultiply(matrix):
    def f(vector):
        return matrix @ vector
    return f

'''
@param x: a real number
@return the sigmoid value
'''
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
'''
@param x: a vector of real numbers
@return a vector where each element is the respective sigmoid value
'''
def sigmoid_v(x):
    return np.vectorize(sigmoid)(x)
'''
@param n: the row dimension
@param m: the column dimension
@return a random n by m matrix
'''
def rand_w(n, m):
    return (.01 - (-.01)) * np.random.random_sample((n, m)) - .01
'''
@param index: an index of a dataframe
@return a randomly shuffled index
'''
def permute(index):
    return random.sample(index, len(index))
'''
@param predicted: a series of predicted values
@param actual: a series of actual target values
@return the mean squared error
'''
def mean_squared_error(predicted, actual):
    p_vec = predicted.to_numpy()
    a_vec = actual.to_numpy()
    diff_vec = p_vec - a_vec
    return math.pow(np.linalg.norm(diff_vec), 2) / len(predicted)
'''
@param data: the data set we are using
@return a function that takes hyperparameters eta and epsilon and an index value that returns a predicted target
'''
def predict_value(data):
    w_init = rand_w(1, data.df.shape[1])
    vec_func = vec(data)
    actual_func = lambda i: data.df.at[i, "Target"]
    def f1(eta, eps):
        def f2(i):
            x = vec_func(i)
            r = actual_func(i)
            def f2_rec(w):
                y = (w @ x.reshape(-1, 1))[0, 0]
                d = r - y
                if abs(d) < eps: return y
                dw = np.vectorize(lambda xj: eta * d * xj)(x)
                return f2_rec(w + dw)
            return f2_rec(w_init)
        return f2
    return f1
'''
@param vec_func: function that takes an index value and gives the sample from the dataset as a vector
@param r: series of actual target values
@param eta: the learning rate
@param index: the index of the data set we are updating
@return ((permuted index, final weight vector), series of predicted target values
'''
def online_update(vec_func, r, eta, index):
    def f(index_remaining, w, y_acc):
        if len(index_remaining) == 0:
            y_idx, y_values = zip(*y_acc)
            return ((permute(index), w), pd.Series(y_values, y_idx))
        else:
            i = index_remaining[0]
            x = vec_func(i)
            yi = (w @ x.reshape(-1, 1))[0, 0]
            dw = np.vectorize(lambda xj: eta * (r[i] - yi) * xj)(x)
            return f(index_remaining[1:], w + dw, y_acc + [(i, yi)])
    return f

'''
@param data: the data set we are using
@param n: the size of the subset of the data we are using
@return a function that takes hyperparameters eta and max error and returns a series of predicted target values
'''
def stochastic_online_gd(data, n):
    w_init = rand_w(1, data.df.shape[1])
    vec_func = vec(data)
    base_index = random.sample(list(data.df.index), k=n)
    r = data.df.loc[base_index, "Target"]
    def f(eta, max_error):
        def epoch(index, start_w):
            return online_update(vec_func, r, eta, index)(index, start_w, [])
        def evaluate(index, w, y = None):
            if y is None or mean_squared_error(y, r) > max_error:
                try:
                    (new_index, final_w), new_y = epoch(index, w)
                    return evaluate(new_index, final_w, new_y)
                except:
                    print("Too Much Recursion!")
                    return y
            else:
                results_df = pd.DataFrame(y)
                results_df["Target"] = r
                print(results_df)
                return y
        return evaluate(base_index, w_init)
    return f










