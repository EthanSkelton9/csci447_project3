import pandas as pd
import math
import numpy as np

class Neural_Net:
    
    # def __init__(data, ):
        
        
        
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
    

    def create_weights(length, nrows):
        matrix = np.random.rand(length, nrows)
        print(matrix)
        return matrix
    
    '''
    create_hidden_nodes() will create the hidden layers and place random weights on them
    
    @returns
    '''
    def create_hidden_nodes(data, vector):
        #start local variabls
        length =  len(vector) #number of hidden layers we want
        nrows = len(data) #number of rows in df
        col = data.columns #columns in df
        #end local variables
        
        if length == 1:
            W1 = Neural_Net.create_weights(vector[0], nrows)
        elif length == 2:
            W1 = Neural_Net.create_weights(vector[0], nrows)
            W2 = Neural_Net.create_weights(vector[1], vector[0])
            pass
        elif length == 3:
            W1 = Neural_Net.create_weights(vector[0], nrows)
            W2 = Neural_Net.create_weights(vector[1], vector[0])
            W3 = Neural_Net.create_weights(vector[2], vector[1])
            pass
        else:
            print("do not have any hidden layers or too many hidden layers")
            
        data_hidden = data
        
        return data_hidden
    
    
    '''
    multi_layer_prop() will create all the matricies that are required for the multi layer propogation to take place
    '''
    def multi_layer_prop(d, vector):
        data = d.df #set the data to the actual dataframe that we want access to
        
        Neural_Net.create_hidden_nodes(data, vector)
        print(data)
        print(vector)
