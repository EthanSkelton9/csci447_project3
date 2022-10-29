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
        v = range(10)
        m = np.matrix([v, v, v])
        x = IF1.value(data)(0)
        print(x)
        print(IF1.sigmoid_v(x))
        print(IF1.matrixmultiply(m)(x))
    def f2():
        DD = DataDictionary()
        data = DD.dataobject(True, "Abalone")
        y = IF1.predict_value(data)(eta = 0.1, eps = 0.01)(2)()
        print(y)
        print(data.df.at[2, "Target"])
    return f2()



def main():
    DD = DataDictionary()
    data = DD.dataobject(True, "Abalone")
    Neural_Net.multi_layer_prop(data, [3])
    #postprocessing ----------------------
if __name__=="__main__":
    #main_Ian()
    main()