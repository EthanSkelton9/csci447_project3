import pandas as pd


'''
Preprocessing class will allow us to create a clean dataset from the raw data that we give it
'''
class Preprocessing:
    '''
    __init__: will initialize the preprocessing class based on the location that the data is
    @param data_loc: the string location where the file we want to read in is
    '''
    def __init__(self, data_loc, columns, target_name):
        self.data_loc = data_loc #set the data location of the class equal to the data location that was sent in
        self.df = None #set the actual data to None for the moment
        self.columns = columns #we need to say what the features are
        self.target_name = target_name
        
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

    def add_column_names(self, df):
        df.columns = self.columns           # Define columns of the data frame with what is given in initialization.
        target_column = df.pop(self.target_name)  # Remove the target column.
        self.features = df.columns                # Define the features of the data by the remaining columns.
        df.insert(len(df.columns), 'Target', target_column)  # Insert the target column at the end.
        self.df = df                              # Define the dataframe for the object.
    
    
    def one_hot(self):
        (features_numerical, features_categorical) = ([], [])
        features_categorical_ohe = []
        for f in self.features:
            try:
                self.df[f].apply(pd.to_numeric)
                features_numerical.append(f)
            except:
                features_categorical.append(f)
                categories = set(self.df[f])
                for cat in categories:
                    features_categorical_ohe.append("{}_{}".format(f, cat))
        self.features_numerical = features_numerical
        self.features_categorical = features_categorical
        one_hot_df = pd.get_dummies(self.df, columns=self.features_categorical)
        self.features_ohe = features_numerical + features_categorical_ohe
        target_column = one_hot_df.pop('Target')
        one_hot_df.insert(len(one_hot_df.columns), 'Target', target_column)
        self.df = one_hot_df
    

    
    
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
    