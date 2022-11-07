from neuralNet import Neural_Net
from preprocessing import Preprocessing
import pandas as pd
from DataDictionary import DataDictionary
import os
import IF1
import numpy as np
from CrossValidation import CrossValidation as CV

class Video():
    
    def performance():
        DD = DataDictionary()
        rdata = DD.dataobject(True, "ForestFires")
        cdata = DD.dataobject(True, "SoyBean")
        cNN = Neural_Net(cdata)
        rNN = Neural_Net(rdata)
        pass
    
    
    
    
    def model(self):
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        DD = DataDictionary()
        rdata = DD.dataobject(True, "ForestFires")
        cdata = DD.dataobject(True, "SoyBean")
        cNN = Neural_Net(cdata)
        rNN = Neural_Net(rdata)
        index = 1
        
        #-------------regression
        nfeatures = len(rNN.data.df.columns)-1
        row = np.delete(rNN.data.df.iloc[index].values, -1)
        
        weights0 = rNN.list_weights(nfeatures, [], 1)
        print("---------------------------No Hidden Layers, Regression---------------------------")
        print("All of the feature values in the node")
        print(row)
        print("======Weights for model with no hidden layers======")
        print("Weights between features and the first class")
        print(weights0[0][0])
        print("=============Output=============")
        print("OutLayer:")
        print(weights0[0]@row)
        
        weights1 = rNN.list_weights(nfeatures, [3], 1)
        print("---------------------------One Hidden Layers, Regression---------------------------")
        print("All of the feature values in the node")
        print(row)
        print("======Weights for model with one hidden layers (features - 1st hidden)======")
        print("Weights between features and the first hidden node")
        print(weights1[0][0])
        print("Weights between features and the second hidden node")
        print(weights1[0][1])
        print("Weights between features and the third hidden node")
        print(weights1[0][2])
        print("======Weights for model with one hidden layers (1st hidden - classification)======")
        print("Weights between 1st hidden layer and the first class")
        print(weights1[1][0])
        
        hidden_layers = rNN.calc_Hidden(weights1, row, 1)
        print("Hidden nodes:")
        print(hidden_layers)
        print("=============Output=============")
        print("OutLayer:")
        print(weights1[1]@hidden_layers[0])
        
        weights2 = rNN.list_weights(nfeatures, [3,4], 1)
        print("---------------------------One Hidden Layers, Regression---------------------------")
        print("All of the feature values in the node")
        print(row)
        print("======Weights for model with one hidden layers (features - 1st hidden)======")
        print("Weights between features and the first hidden node")
        print(weights2[0][0])
        print("Weights between features and the second hidden node")
        print(weights2[0][1])
        print("Weights between features and the third hidden node")
        print(weights2[0][2])
        print("======Weights for model with one hidden layers (1st hidden - 2nd hidden)======")
        print("Weights between first hidden layer and the first hidden node")
        print(weights2[1][0])
        print("Weights between fist hidden layer and the second hidden node")
        print(weights2[1][1])
        print("======Weights for model with one hidden layers (2st hidden - classification)======")
        print("Weights between 2st hidden layer and the first class")
        print(weights2[2][0])
        
        hidden_layers = rNN.calc_Hidden(weights2, row, 2)
        print("First Hidden Layer")
        print(hidden_layers[0])
        print("Second Hidden Layer")
        print(hidden_layers[1])
        print("=============Output=============")
        print("OutLayer:")
        print(weights2[2]@hidden_layers[1])
        
        
        #-------------classification
        nfeatures = len(cNN.data.df.columns)-1
        target_length = len(cNN.data.classes)
        row = np.delete(cNN.data.df.iloc[index].values, -1)
        
        weights0 = cNN.list_weights(nfeatures, [], target_length)
        print("---------------------------No Hidden Layers, Classification---------------------------")
        print("All of the feature values in the node")
        print(row)
        print("======Weights for model with no hidden layers======")
        print("Weights between features and the first class")
        print(weights0[0][0])
        print("Weights between features and the second class")
        print(weights0[0][1])
        print("Weights between features and the third class")
        print(weights0[0][2])
        print("Weights between features and the fourth class")
        print(weights0[0][3])
        print("=============Output=============")
        print("OutLayer:")
        print(weights0[0]@row)
        
        
        weights1 = cNN.list_weights(nfeatures, [3], target_length)
        print("---------------------------One Hidden Layers, Classification---------------------------")
        print("All of the feature values in the node")
        print(row)
        print("======Weights for model with one hidden layers (features - 1st hidden)======")
        print("Weights between features and the first hidden node")
        print(weights1[0][0])
        print("Weights between features and the second hidden node")
        print(weights1[0][1])
        print("Weights between features and the third hidden node")
        print(weights1[0][2])
        print("======Weights for model with one hidden layers (1st hidden - classification)======")
        print("Weights between 1st hidden layer and the first class")
        print(weights1[1][0])
        print("Weights between 1st hidden layer and the second class")
        print(weights1[1][1])
        print("Weights between 1st hidden layer and the third class")
        print(weights1[1][2])
        print("Weights between 1st hidden layer and the fourth class")
        print(weights1[1][3])
        
        hidden_layers = cNN.calc_Hidden(weights1, row, 1)
        print("Hidden nodes:")
        print(hidden_layers)
        print("=============Output=============")
        print("OutLayer:")
        print(weights1[1]@hidden_layers[0])
        
        
        weights2 = cNN.list_weights(nfeatures, [3,2], target_length)
        print("---------------------------One Hidden Layers, Classification---------------------------")
        print("All of the feature values in the node")
        print(row)
        print("======Weights for model with one hidden layers (features - 1st hidden)======")
        print("Weights between features and the first hidden node")
        print(weights2[0][0])
        print("Weights between features and the second hidden node")
        print(weights2[0][1])
        print("Weights between features and the third hidden node")
        print(weights2[0][2])
        print("======Weights for model with one hidden layers (1st hidden - 2nd hidden)======")
        print("Weights between first hidden layer and the first hidden node")
        print(weights2[1][0])
        print("Weights between fist hidden layer and the second hidden node")
        print(weights2[1][1])
        print("======Weights for model with one hidden layers (2st hidden - classification)======")
        print("Weights between 2st hidden layer and the first class")
        print(weights2[2][0])
        print("Weights between 2st hidden layer and the second class")
        print(weights2[2][1])
        print("Weights between 2st hidden layer and the third class")
        print(weights2[2][2])
        print("Weights between 2st hidden layer and the fourth class")
        print(weights2[2][3])
        
        hidden_layers = cNN.calc_Hidden(weights2, row, 2)
        print("First Hidden Layer")
        print(hidden_layers[0])
        print("Second Hidden Layer")
        print(hidden_layers[1])
        print("=============Output=============")
        print("OutLayer:")
        print(weights2[2]@hidden_layers[1])
        
    
    
    
    
      
    def propagation(self):
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("---------------Propagating through the Soybeans dataset---------------------")
        DD = DataDictionary()
        data = DD.dataobject(True, "SoyBean")
        NN = Neural_Net(data)
        num_hidden = 2
        
        nfeatures = len(NN.data.df.columns)-1
        target_length = len(NN.data.classes)
        row = np.delete(NN.data.df.iloc[1].values, -1)
        
        weights = NN.list_weights(nfeatures, [3,4], target_length)
        hidden_layers = []
        layers = []
        
        layers.append(row)
        
        if(num_hidden == 0): #if there are no hidden layers
            hidden_layers
        else:
            print("++++++++++++++++++++++++++++")
            print("List of features")
            print(row)
            print("First weight matrix")
            print(weights[0])
            print("Multiply the first weight matrix by the features to get the first hidden layer")
            hidden_layers.append(NN.sigmoid_v(weights[0]@row)) #find the first hidden layer
            print(hidden_layers[0])
            print("++++++++++++++++++++++++++++")
            for i in range(num_hidden-1):
                print("++++++++++++++++++++++++++++")
                print("Second weight matrix")
                print(weights[i+1])
                hidden_layers[i]
                hidden_layers.append(NN.sigmoid_v(weights[i+1]@hidden_layers[i])) #all hidden layers after
        print("Multiply the second matrix by the first hidden layer to get")
        print("Second hidden layer:")
        print(hidden_layers[1])
        print("++++++++++++++++++++++++++++")
        print("====Output=====")
        print("Final weight matrix")
        print(weights[2])
        print("Multiply the final weight matrix by the last hidden layer to get the output")
        print(weights[2]@hidden_layers[1])
    
    
    
    
    
    def grad_calc():
        DD = DataDictionary()
        rdata = DD.dataobject(True, "ForestFires")
        cdata = DD.dataobject(True, "SoyBean")
        cNN = Neural_Net(cdata)
        rNN = Neural_Net(rdata)
        pass
    
    def weight_update():
        DD = DataDictionary()
        rdata = DD.dataobject(True, "ForestFires")
        cdata = DD.dataobject(True, "SoyBean")
        cNN = Neural_Net(cdata)
        rNN = Neural_Net(rdata)
        pass
    
    
    
    
    
    def average_performance(self):
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        DD = DataDictionary()
        rdata = DD.dataobject(True, "ForestFires")
        cdata = DD.dataobject(True, "SoyBean")
        cNN = Neural_Net(cdata)
        rNN = Neural_Net(rdata)
        
        #regression ---------------------------------
        print("==============================Regression 10-fold==============================")
        csv = os.getcwd() + '\\' + str(rNN.data) + '\\' + "{}_Analysis_0.csv".format(str(rNN.data))
        analysis0 = pd.read_csv(csv, index_col=0)
        csv = os.getcwd() + '\\' + str(rNN.data) + '\\' + "{}_Analysis_1.csv".format(str(rNN.data))
        analysis1 = pd.read_csv(csv, index_col=0)
        csv = os.getcwd() + '\\' + str(rNN.data) + '\\' + "{}_Analysis_2.csv".format(str(rNN.data))
        analysis2 = pd.read_csv(csv, index_col=0)
        print("+++++++++++++++++++++++Analysis for 0 Hidden Layers+++++++++++++++++++++++")
        print(analysis0)
        print("+++++++++++++++++++++++Analysis for 1 Hidden Layer+++++++++++++++++++++++")
        print(analysis1)
        print("+++++++++++++++++++++++Analysis for 2 Hidden Layers+++++++++++++++++++++++")
        print(analysis2)
        
        #classification -----------------------------
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("==============================Classification 10-fold==============================")
        csv = os.getcwd() + '\\' + str(cNN.data) + '\\' + "{}_Analysis_0.csv".format(str(cNN.data))
        analysis0 = pd.read_csv(csv, index_col=0)
        csv = os.getcwd() + '\\' + str(cNN.data) + '\\' + "{}_Analysis_1.csv".format(str(cNN.data))
        analysis1 = pd.read_csv(csv, index_col=0)
        csv = os.getcwd() + '\\' + str(cNN.data) + '\\' + "{}_Analysis_2.csv".format(str(cNN.data))
        analysis2 = pd.read_csv(csv, index_col=0)
        print("+++++++++++++++++++++++Analysis for 0 Hidden Layers+++++++++++++++++++++++")
        print(analysis0)
        print("+++++++++++++++++++++++Analysis for 1 Hidden Layer+++++++++++++++++++++++")
        print(analysis1)
        print("+++++++++++++++++++++++Analysis for 2 Hidden Layers+++++++++++++++++++++++")
        print(analysis2)