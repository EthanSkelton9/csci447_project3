from preprocessing import Preprocessing
import pandas as pd
from DataDictionary import DataDictionary
import os

def main_Ian():
    DD = DataDictionary()
    for name in DD.datanames:
        data = Preprocessing(*DD.data(name))
        data.add_column_names(pd.read_csv(data.data_loc, header = None))
        data.df.to_csv(os.getcwd() + '\\' + name + '\\' + "{}_w_colnames.csv".format(name))
        data.one_hot()
        data.df.to_csv(os.getcwd() + '\\' + name + '\\' + "{}_onehot.csv".format(name))
        print(name)
        print("Numerical: {}".format(data.features_numerical))
        print("Categorical: {}".format(data.features_categorical))



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