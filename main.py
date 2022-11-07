from neuralNet import Neural_Net
from preprocessing import Preprocessing
import pandas as pd
from DataDictionary import DataDictionary
import os
import IF1
import numpy as np
from CrossValidation import CrossValidation as CV

def main_Ian():
    def f1():
        DD = DataDictionary()
        data = DD.dataobject(True, "SoyBean")
        NN = Neural_Net(data)
        y = NN.stochastic_online_gd()(eta=0.1, max_error=1, hidden_vector = [8, 4], alpha = 0.9)
    def f2():
        DD = DataDictionary()
        data = DD.dataobject(True, "SoyBean")
        DataCV = CV(data)
        print(DataCV.training_test_dicts(data.df))
    return f2()



def main():
    
    DD = DataDictionary()
    data = DD.dataobject(True, "SoyBean")
    SoyBeans = Neural_Net(data)
    SoyBeans.multi_layer_prop([1,3,4], classification = True)
    
    
if __name__=="__main__":
    main_Ian()
    #main()
    #y = NN.tuning([3,1,1])