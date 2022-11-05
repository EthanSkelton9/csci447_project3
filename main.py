from neuralNet import Neural_Net
from preprocessing import Preprocessing
import pandas as pd
from DataDictionary import DataDictionary
import os
import IF1
import numpy as np

def main_Ian():
    def f1():
        DD = DataDictionary()
        data = DD.dataobject(True, "Abalone")
        v = range(11)
        m = np.matrix([v, v, v])
        x = IF1.vec(data)(0)
        print(x)
        print(IF1.sigmoid_v(x))
        print(IF1.matrixmultiply(m)(x))
    def f2():
        DD = DataDictionary()
        data = DD.dataobject(True, "Abalone")
        y = IF1.predict_value(data)(eta = 0.1, eps = 0.01)(2)
        print(y)
        print(data.df.at[2, "Target"])
    def f3():
        DD = DataDictionary()
        data = DD.dataobject(True, "Abalone")
        y = IF1.stochastic_online_gd(data, 40)(eta = 0.1, max_error = 10)
    def f4():
        DD = DataDictionary()
        data = DD.dataobject(True, "SoyBean")
        NN = Neural_Net(data)
        y = NN.stochastic_online_gd(data, 20)(eta=0.1, max_error=15, hidden_vector = [8, 4])
    return f4()



def main():
    
    DD = DataDictionary()
    data = DD.dataobject(True, "SoyBean")
    SoyBeans = Neural_Net(data)
    SoyBeans.multi_layer_prop([1,3,4], classification = True)
    
    
if __name__=="__main__":
    main_Ian()
    #main()