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
        y = IF1.predict_value(data)(eta = 0.1, eps = 0.01)(2)
        print(y)
        print(data.df.at[2, "Target"])
    def f3():
        DD = DataDictionary()
        data = DD.dataobject(True, "Abalone")
        y = IF1.stochastic_online_gd(data, 30)(eta = 0.1, error_max = 5)
    return f3()



def main():
    pass
    #postprocessing ----------------------
if __name__=="__main__":
    main_Ian()
    #main()