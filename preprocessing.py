import pandas as pd


'''
Preprocessing class will allow us to create a clean dataset from the raw data that we give it
'''
class Preprocessing:
    '''
    __init__: will initialize the preprocessing class based on the location that the data is
    @param data_loc: the string location where the file we want to read in is
    '''
    def __init__(self, data_loc, features):
        self.data_loc = data_loc #set the data location of the class equal to the data location that was sent in
        self.df = None #set the actual data to None for the moment
        self.features = features #we need to say what the features are
        
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
        (features_numerical, features_categorical) = ([], [])
        features_categorical_ohe = []
        for f in data.features:
            try:
                df[f].apply(pd.to_numeric)
                features_numerical.append(f)
            except:
                features_categorical.append(f)
                categories = set(df[f])
                for cat in categories:
                    features_categorical_ohe.append("{}_{}".format(f, cat))
        data.features_numerical = features_numerical
        data.features_categorical = features_categorical
        one_hot_df = pd.get_dummies(df, columns=data.features_categorical)
        self.features_ohe = features_numerical + features_categorical_ohe
        target_column = one_hot_df.pop('Target')
        one_hot_df.insert(len(one_hot_df.columns), 'Target', target_column)
        return one_hot_df
    
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
    