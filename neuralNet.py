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

    # def sigmoid_v(x):
    #     return np.vectorize(sigmoid)(x)

    def rand_w(n, m):
        return (.01 - (-.01)) * np.random.random_sample((n, m)) - .01
    
    '''
    create_weights() - creates a matrix of weights randomly
    @param length - length of hidden layers we want
    @nrow - number of rows that we have before the hidden layer
    @return - a matrix of weights
    '''
    def create_weights(length, nrows):
        matrix = np.random.rand(length, nrows)
        return matrix
    
    '''
    all_weights() - creates a list of all the weight matrix
    @param - length - number of hiden layers that we want
    @nrows - number of rows that are in the dataset
    @vector - the vector of hidden layers
    @return weights - list of the weights that are calculated
    '''
    def all_weights(length, nrows, vector):
        weights = []
        
        
        if length == 1: #one hidden layer
            W1 = Neural_Net.create_weights(vector[0], nrows) 
            weights = [W1]
        elif length == 2: #two hidden layers
            W1 = Neural_Net.create_weights(vector[0], nrows)
            W2 = Neural_Net.create_weights(vector[1], vector[0])
            weights = [W1, W2]
        elif length == 3: #three hidden layers
            W1 = Neural_Net.create_weights(vector[0], nrows)
            W2 = Neural_Net.create_weights(vector[1], vector[0])
            W3 = Neural_Net.create_weights(vector[2], vector[1])
            weights = [W1, W2, W3]
        else:
            print("do not have any hidden layers or too many hidden layers")
            
        return weights
    
    '''
    calc_Hidden - calculates the hidden layers on the weights
    @param weights[] - weights  between the layers 
    @param data - the data that we want to read in
    @length - number of hidden layers that we have
    
    @return hidden_layers - returns the hidden layers that we created

    '''
    
    def calc_Hidden(weights, data, length):
        new_data = data.copy() #copy of data
        del new_data['Target'] # remove the target column
        hidden_layers = []
        
        if length == 1: #if there is only one hidden layer
            hidden_layers.append(weights[0] @ new_data) #create hidden layers for first layer
        elif length == 2: #if there is two hidden layer
            hidden_layers.append(weights[0] @ new_data) #create hidden layers for first layer
            hidden_layers.append(weights[1] @ hidden_layers[0]) #create hidden layer between first and second
            pass
        elif length == 3: #if there is three hidden layer
            hidden_layers.append(weights[0] @ new_data) #create hidden layers for first layer
            hidden_layers.append(weights[1] @ hidden_layers[0]) #create hidden layer between first and second
            hidden_layers.append(weights[2] @ hidden_layers[1]) #create hidden layer between second and thrid
            pass
        else:
            print("do not have any hidden layers or too many hidden layers")
            
        return hidden_layers #return the hidden layers
                
        
        
             
    '''
    create_hidden_nodes() will create the hidden layers and place random weights on them
    @param data - the data that we are reading in
    @param vector - the hidden layers that we are reading in
    
    @returns hidden layers
    '''
    def create_hidden_nodes(data, vector):
        #start local variabls
        length =  len(vector) #number of hidden layers we want
        nrows = len(data) #number of rows in df
        col = data.columns #columns in df
        #end local variables
        
        weights = Neural_Net.all_weights(length, nrows, vector) #create the weights for each layer 
        hidden = Neural_Net.calc_Hidden(weights, data, length) #create the hidden nodes 
        
        return hidden
    
    
    '''
    multi_layer_prop() will create all the matricies that are required for the multi layer propogation to take place
    '''
    def multi_layer_prop(d, vector):
        data = d.df #set the data to the actual dataframe that we want access to
        
        h = Neural_Net.create_hidden_nodes(data, vector) #creates the hidden nodes 
        print(h)
        return h
