from preprocessing import Preprocessing
import pandas as pd
from DataDictionary import DataDictionary
import os

def main_Ian():
    DD = DataDictionary()
    data = DD.dataobject(True, "Abalone")
    print(data.df.head(10))
    dataobjects = DD.dataobjects(True)
    print(dataobjects)
    for name in DD.datanames:
        print(dataobjects[name].df.head(10))



def main():
    pass
    #postprocessing ----------------------
if __name__=="__main__":
    main_Ian()
    #main()