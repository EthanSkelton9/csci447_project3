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
    #preprocessing ---------------------
    
    #glass
    # g = Preprocessing("raw_data/glass.csv")
    # g.preprocess()
    
    #soybeans
    
    s = Preprocessing("raw_data/soybean-small.csv")
    s.readcsv()
    clean = s.preprocess(None, None)
    
    
    #breast_cancer
    
    # b = Preprocessing("raw_data/breast-cancer-wisconsin.csv")
    # b.preprocess(None, '3')
    
    #abalone
    
    # a = Preprocessing("raw_data/abalone.csv")
    # a.preprocess()
    
    #forestfires
    
    # f = Preprocessing("raw_data/forestfires.csv")
    # f.preprocess()
    
    #machine
    
    # m = Preprocessing("raw_data/machine.csv")
    # m.preprocess()
    
    
    #postprocessing ----------------------
if __name__=="__main__":
    main_Ian()
    #main()