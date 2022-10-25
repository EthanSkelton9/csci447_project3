from preprocessing import Preprocessing
import pandas as pd
from functools import partial as pf

class DataDictionary:
    def __init__(self):
        self.datanames = ["Abalone",
                          "BreastCancer",
                          "ForestFires",
                          "Glass",
                          "Hardware",
                          "SoyBean"]

    def dataobjects(self, preprocessed, names = None):
        if names == None: names = self.datanames
        return pd.Series(names, index=names).map(pf(self.dataobject, preprocessed)) #for each name give the data object

    def dataobject(self, preprocessed, name):
        data = Preprocessing(*self.metadata(name))
        if preprocessed:
            data.set_to_raw_data()   #give it the raw data
            data.add_column_names()  #add column names as well as define features
            data.clean_missing()
            data.one_hot()           #implement one hot encoding
            data.z_score_normalize() #normalize data with z score
        return data

    def metadata(self, name):
        if name == "Abalone": return self.abalone()
        if name == "BreastCancer": return self.breastcancer()
        if name == "ForestFires": return self.forestfires()
        if name == "Glass": return self.glass()
        if name == "Hardware": return self.hardware()
        if name == "SoyBean": return self.soybean()


    def abalone(self):
        name = "Abalone"
        file = 'raw_data/abalone.csv'
        columns = ['Sex',  # For Abalone
         'Length',
         'Diameter',
         'Height',
         'Whole Weight',
         'Shucked Weight',
         'Viscera Weight',
         'Shell Weight',
         'Rings' #Target
         ]
        replace = None
        target_name = 'Rings'
        return (name, file, columns, target_name, replace)

    def breastcancer(self):
        name = "BreastCancer"
        file = 'raw_data/breast-cancer-wisconsin.csv'
        columns = [   'Id',   # For Breast Cancer
            'Clump Thickness',
            'Uniformity of Cell Size',
            'Uniformity of Cell Shape',
            'Marginal Adhesion',
            'Single Epithelial Cell Size',
            'Bare Nuclei',
            'Bland Chromatin',
            'Normal Nucleoli',
            'Mitoses',
            'Class'  #Target
        ]
        target_name = 'Class'
        replace = '3'
        return (name, file, columns, target_name, replace)

    def forestfires(self):
        name = "ForestFires"
        file = 'raw_data/forestfires.csv'
        columns = [ 'X', # For Forest Fires
          'Y',
          'Month',
          'Day',
          'FFMC',
          'DMC',
          'DC',
          'ISI',
          'Temp',
          'RH',
          'Wind',
          'Rain',
          'Area'  #Target
        ]
        replace = None
        target_name = 'Area'
        return (name, file, columns, target_name, replace)

    def glass(self):
        name = "Glass"
        file = 'raw_data/glass.csv'
        columns = [   "Id number: 1 to 214",  # For Glass
            "RI: refractive index",
            "Na: Sodium",
            "Mg: Magnesium",
            "Al: Aluminum",
            "Si: Silicon",
            "K: Potassium",
            "Ca: Calcium",
            "Ba: Barium",
            "Fe: Iron",
            "Class" #Target
        ]
        target_name = 'Class'
        replace = None
        return (name, file, columns, target_name, replace)

    def hardware(self):
        name = "Hardware"
        file = 'raw_data/machine.csv'
        columns = [   "Vendor Name",  # For Computer Hardware
            "Model Name",
            "MYCT",
            "MMIN",
            "MMAX",
            "CACH",
            "CHMIN",
            "CHMAX",
            "PRP",  #Target
            "ERP"
        ]
        target_name = 'PRP'
        replace = None
        return (name, file, columns, target_name, replace)

    def soybean(self):
        name = "SoyBean"
        file = 'raw_data/soybean-small.csv'
        columns =  ['Date',  # For Soy Bean
         'Plant-Stand',
         'Precip',
         'Temp',
         'Hail',
         'Crop-Hist',
         'Area-Damaged',
         'Severity',
         'Seed-TMT',
         'Germination',
         'Plant-Growth',
         'Leaves',
         'Leafspots-Halo',
         'Leafspots-Marg',
         'Leafspot-Size',
         'Leaf-Shread',
         'Leaf-Malf',
         'Leaf-Mild',
         'Stem',
         'Lodging',
         'Stem-Cankers',
         'Canker-Lesion',
         'Fruiting-Bodies',
         'External Decay',
         'Mycelium',
         'Int-Discolor',
         'Sclerotia',
         'Fruit-Pods',
         'Fruit Spots',
         'Seed',
         'Mold-Growth',
         'Seed-Discolor',
         'Seed-Size',
         'Shriveling',
         'Roots',
         'Class'  #Target
         ]
        replace = None
        target_name = 'Class'
        return (name, file, columns, target_name, replace)