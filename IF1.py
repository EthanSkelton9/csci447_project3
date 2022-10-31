import pandas as pd
import math
import numpy as np
import functools
import random

'''
@param data: the data set we are using
@return: function that takes a row of the data and returns its numpy vector
'''
def vec(data):
    '''
    @param i: index of the data set
    @return: numpy row vector of the data with a 1 appended at the beginning for the bias weight
    '''
    def f(i):
        return np.concatenate(([1], data.df.loc[i, data.features_ohe]))
    return f

'''
@param matrix: the matrix used to multiply
@return function that transforms vectors into vectors by multiplying by the given matrix
'''
def matrixmultiply(matrix):
    '''
    @param vector: the numpy vector that is multiplied
    @return: numpy column vector that is the product of the matrix multiplication
    '''
    def f(vector):
        return matrix @ vector.reshape(-1, 1)
    return f

'''
@param x: a real number
@return: the sigmoid value
'''
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
'''
@param x: a vector of real numbers
@return: a vector where each element is the respective sigmoid value
'''
def sigmoid_v(x):
    return np.vectorize(sigmoid)(x)
'''
@param n: the row dimension
@param m: the column dimension
@return: a random n by m matrix
'''
def rand_w(n, m):
    return (.01 - (-.01)) * np.random.random_sample((n, m)) - .01
'''
@param index: an index of a dataframe
@return: a randomly shuffled index
'''
def permute(index):
    return random.sample(index, len(index))
'''
@param predicted: a series of predicted values
@param actual: a series of actual target values
@return: the mean squared error
'''
def mean_squared_error(predicted, actual):
    return math.pow(np.linalg.norm(predicted.to_numpy() - actual.to_numpy()), 2) / len(predicted)
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
@return ((permuted index, final weight vector), series of predicted target values)
@used in: stochastic_online_gd
'''
def online_update(vec_func, r, eta, index):
    '''
    @param index_remaining: index left to iterate through
    @param w: the current weight matrix
    @param y_acc: the current set of accumulated predictions
    @return: new index, final weight matrix, and complete set of predictions after iterated through index
    '''
    def f(index_remaining, w, y_acc):
        if len(index_remaining) == 0:                                      #if there is nothing more to iterate through
            y_idx, y_values = zip(*y_acc)                                  #unzip to get y index and y values
            return (permute(index), w, pd.Series(y_values, y_idx))
        else:
            i = index_remaining[0]                                         #the next index value
            x = vec_func(i)                                                #the next sample vector
            yi = (w @ x.reshape(-1, 1))[0, 0]                              #the new y value
            dw = np.vectorize(lambda xj: eta * (r[i] - yi) * xj)(x)        #the gradient of the weights
            return f(index_remaining[1:], w + dw, y_acc + [(i, yi)])
    return f

'''
@param data: the data set we are using
@param n: the size of the subset of the data we are using
@return a function that takes hyperparameters eta and max error and returns a series of predicted target values
'''
def stochastic_online_gd(data, n):
    w_init = rand_w(1, data.df.shape[1])                                        # initial randomized weights
    vec_func = vec(data)                                                        # create vector function for data
    base_index = random.sample(list(data.df.index), k=n)                        # create a shuffled index for iteration
    r = data.df.loc[base_index, "Target"]                                       # column of target values
    '''
    @param eta: the learning rate
    @param max_error: the maximum tolerance used
    @return: function that uses the hyperparameters to return a series of predicted values
    '''
    def f(eta, max_error):
        '''
        @param index: the index to iterate through
        @param start_w: the starting weight matrix to use for the epoch
        @return: new permuted index, weight matrix learned from data, and a series of predicted values
        '''
        def epoch(index, start_w):
            return online_update(vec_func, r, eta, index)(index, start_w, [])
        '''
        @param index: index after an epoch or starting the first epoch
        @param w: weight matrix after an epoch or starting the first epoch
        @param y: current prediction
        @return: the final prediction 
        '''
        def evaluate(index, w, y = None):
            if y is None or mean_squared_error(y, r) > max_error:        #if the predictions have not converged yet
                try:
                    new_index, final_w, new_y = epoch(index, w)          #run through another epoch
                    return evaluate(new_index, final_w, new_y)           #evaluate to see if there is convergence
                except:
                    print("Too Much Recursion!")
                    return y                                        #return the last prediction before recursion error
            else:
                results_df = pd.DataFrame(y)
                results_df["Target"] = r
                print(results_df)
                return y                                            #return final prediction
        return evaluate(base_index, w_init)
    return f










