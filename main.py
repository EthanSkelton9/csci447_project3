from preprocessing import Preprocessing

def main_Ian():
    s = Preprocessing("raw_data/soybean-small.csv")


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
