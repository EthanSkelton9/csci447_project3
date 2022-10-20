import pandas as pd


'''
Preprocessing class will allow us to create a clean dataset from the raw data that we give it
'''
class Preprocessing:
    '''
    __init__: will initialize the preprocessing class based on the location that the data is
    @param data_loc: the string location where the file we want to read in is
    '''
    def __init__(self, data_loc):
        self.data_loc = data_loc #set the data location of the class equal to the data location that was sent in
        self.df = None #set the actual data to None for the moment
        
    '''
    readcsv: will take the location of the file and convert it to pandas
    @return self.data - panda df of the csv for us to work with
    '''   
    def readcsv(self):
        dn = self.data_loc
        self.df = pd.read_csv(dn) #read the data location to pandas dataframe
        return self.df
    
    '''
    clean_missing: removes '?' that are in the dataframe and replaces them with another value
    @param replace: the value to replace the missing value with
    '''
    def clean_missing(data, replace):
        for col_name in data.df.columns:
            data.df[col_name] = data.df[col_name].replace(['?'], [replace])
        return data
        
    def z_score(data):
        pass
    
    
    def one_hot(data):
        pass
    
    def column_names(data):
        pass 
    
    
    '''
    preprocess: will preprocess the data for a given file and return the cleaned dataframe 
    @param features - the features that will need to be read in to clean the df
    @param missing - if there are missing values in the dataframe then replace with this value
    @return clean - the cleaned datafram
    '''
    def preprocess(self, features, missing):
        data = self.readcsv() #reads in csv to panda
        if features == None:
            pass
        if missing != None:
            self.clean_missing(data, missing)
        clean = self.readcsv()
        return clean
    