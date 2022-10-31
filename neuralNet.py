import pandas as pd
import math
import numpy as np
import IF1
import random

class Neural_Net:

    def __init__(self, data):
        self.data = data

    '''
    @param data: the data set we are using
    @return: function that takes a row of the data and returns its numpy vector
    '''
    def vec(self, data):
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
    def matrixmultiply(self, matrix):
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
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    '''
    @param x: a vector of real numbers
    @return: a vector where each element is the respective sigmoid value
    '''
    def sigmoid_v(self, x):
        return np.vectorize(self.sigmoid)(x)

    '''
    @param n: the row dimension
    @param m: the column dimension
    @return: a random n by m matrix
    '''
    def rand_w(self, n, m):
        return (.01 - (-.01)) * np.random.random_sample((n, m)) - .01

    '''
    @param index: an index of a dataframe
    @return: a randomly shuffled index
    '''
    def permute(self, index):
        return random.sample(index, len(index))


    '''
    @param predicted: a series of predicted values
    @param actual: a series of actual target values
    @return: the mean squared error
    '''
    def mean_squared_error(self, predicted, actual):
        return math.pow(np.linalg.norm(predicted.to_numpy() - actual.to_numpy()), 2) / len(predicted)

    '''
    @param vec_func: function that takes an index value and gives the sample from the dataset as a vector
    @param r: series of actual target values
    @param eta: the learning rate
    @param index: the index of the data set we are updating
    @return ((permuted index, final weight vector), series of predicted target values)
    @used in: stochastic_online_gd
    '''

    def online_update(self, vec_func, r, eta, index):
        '''
        @param index_remaining: index left to iterate through
        @param w: the current weight matrix
        @param y_acc: the current set of accumulated predictions
        @return: new index, final weight matrix, and complete set of predictions after iterated through index
        '''
        def f(index_remaining, w, y_acc):
            if len(index_remaining) == 0:  # if there is nothing more to iterate through
                y_idx, y_values = zip(*y_acc)  # unzip to get y index and y values
                return (self.permute(index), w, pd.Series(y_values, y_idx))
            else:
                i = index_remaining[0]  # the next index value
                x = vec_func(i)  # the next sample vector
                yi = (w @ x.reshape(-1, 1))[0, 0]  # the new y value
                dw = eta * (r[i] - yi) * x  # the gradient of the weights
                return f(index_remaining[1:], w + dw, y_acc + [(i, yi)])

        return f

    '''
    @param data: the data set we are using
    @param n: the size of the subset of the data we are using
    @return a function that takes hyperparameters eta and max error and returns a series of predicted target values
    '''

    def stochastic_online_gd(self, data, n):
        w_init = self.rand_w(1, data.df.shape[1])  # initial randomized weights
        vec_func = self.vec(data)  # create vector function for data
        base_index = random.sample(list(data.df.index), k=n)  # create a shuffled index for iteration
        r = data.df.loc[base_index, "Target"]  # column of target values
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
                return self.online_update(vec_func, r, eta, index)(index, start_w, [])

            '''
            @param index: index after an epoch or starting the first epoch
            @param w: weight matrix after an epoch or starting the first epoch
            @param y: current prediction
            @return: the final prediction 
            '''

            def evaluate(index, w, y=None):
                if y is None or self.mean_squared_error(y, r) > max_error:  # if the predictions have not converged yet
                    try:
                        new_index, final_w, new_y = epoch(index, w)  # run through another epoch
                        return evaluate(new_index, final_w, new_y)  # evaluate to see if there is convergence
                    except:
                        print("Too Much Recursion!")
                        return y  # return the last prediction before recursion error
                else:
                    results_df = pd.DataFrame(y)
                    results_df["Target"] = r
                    print(results_df)
                    return y  # return final prediction

            return evaluate(base_index, w_init)

        return f

    #------------------------------------------------------------------------------------------------------
    
    '''
    create_weights() - creates a matrix of weights randomly
    @param length: length of hidden layers we want
    @param nrow: number of rows that we have before the hidden layer
    @return: a matrix of weights with dimensions length and nrows
    '''
    def create_rand_weights(nrows, length):
        matrix = np.random.rand(length, nrows)
        return matrix
    
    '''
    all_weights() - creates a list of all the weight matrix
    @param - length - number of hiden layers that we want
    @nrows - number of rows that are in the dataset
    @vector - the vector of hidden layers
    @return weights - list of the weights that are calculated
    '''
    def list_weights(nrows, vector, length, target_len):
        weights = [] #list of weight matrices that we will be returning
        
        if length == 1: #one hidden layer
            W1 = Neural_Net.create_rand_weights(nrows, vector[0])
            W2 = Neural_Net.create_rand_weights(vector[0], target_len) #class assignment
            weights = [W1, W2]
        elif length == 2: #two hidden layers
            W1 = Neural_Net.create_rand_weights(nrows, vector[0])
            W2 = Neural_Net.create_rand_weights(vector[0], vector[1])
            W3 = Neural_Net.create_rand_weights(vector[1], target_len) #class assignment
            weights = [W1, W2, W3]
        elif length == 3: #three hidden layers
            W1 = Neural_Net.create_rand_weights(nrows, vector[0])
            W2 = Neural_Net.create_rand_weights(vector[0], vector[1])
            W3 = Neural_Net.create_rand_weights(vector[1], vector[2])
            W4 = Neural_Net.create_rand_weights(vector[2], target_len) #class assignment
            weights = [W1, W2, W3, W4]
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
            hidden_layers.append(IF1.sigmoid_v(weights[0] @ new_data)) #create hidden layers for first layer
        elif length == 2: #if there is two hidden layer
            hidden_layers.append(IF1.sigmoid_v(weights[0] @ new_data)) #create hidden layers for first layer
            hidden_layers.append(IF1.sigmoid_v(weights[1] @ hidden_layers[0])) #create hidden layer between first and second
            pass
        elif length == 3: #if there is three hidden layer
            hidden_layers.append(IF1.sigmoid_v(weights[0] @ new_data)) #create hidden layers for first layer
            hidden_layers.append(IF1.sigmoid_v(weights[1] @ hidden_layers[0])) #create hidden layer between first and second
            hidden_layers.append(IF1.sigmoid_v(weights[2] @ hidden_layers[1])) #create hidden layer between second and thrid
            pass
        else:
            print("do not have any hidden layers or too many hidden layers")
            
        return hidden_layers #return the hidden layers
    
    '''
    multi_layer_prop() will create all the matricies that are required for the multi layer propogation to take place
    '''
    def multi_layer_prop(d, vector, classification):
        
         #start local variabls
        data = d.df #set the data to the actual dataframe that we want access to
        length =  len(vector) #number of hidden layers we want
        nrows = len(data) #number of rows in df
        col = data.columns #columns in df
        target = None #target classes that we are going for
        target_len = 1 #target length set to 1 if regression
        
        if classification:
            target = data["Target"].unique() #set target to each target value if classs
            target_len = len(target) #set target length
            
        #end local variables
        
        weights = Neural_Net.list_weights(nrows, vector, length, target_len) #create the weights for each layer 
        hidden = Neural_Net.calc_Hidden(weights, data, length) #create the hidden nodes 
        
        print(weights[3])
        
        return hidden
