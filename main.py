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
        print(IF1.error(data)(0))
    return f2()



def main():
    pass
    #postprocessing ----------------------
if __name__=="__main__":
    main_Ian()
    #main()