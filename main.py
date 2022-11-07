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
        y = NN.stochastic_online_gd()(eta=0.1, hidden_vector = [8, 4], alpha = 0.9)
        print(y)
    def f2():
        DD = DataDictionary()
        data = DD.dataobject(True, "Abalone")
        DataCV = CV(data)
        # DataCV.test(eta_space = np.linspace(0.1, 0.3, 3), alpha_space = [0, 0.8, 0.9], new=False, appendCount=3)
        DataCV.analysisFunction()
    def f3():
        DD = DataDictionary()
        data = DD.dataobject(True, "ForestFires")
        DataCV = CV(data)
        DataCV.latex_display()
    return f3()



def main():
    
    DD = DataDictionary()
    data = DD.dataobject(True, "SoyBean")
    SoyBeans = Neural_Net(data)
    SoyBeans.multi_layer_prop([1,3,4], classification = True)
    
    
if __name__=="__main__":
    main_Ian()
    #main()
    