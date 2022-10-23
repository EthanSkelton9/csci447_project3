

class DataDictionary:
    def __init__(self):
        self.datanames = ["Abalone",
                          "BreastCancer",
                          "ForestFires",
                          "Glass",
                          "Hardware",
                          "SoyBean"]

    def data(self, name):
        if name == "Abalone": return self.abalone()
        if name == "BreastCancer": return self.breastcancer()
        if name == "ForestFires": return self.forestfires()
        if name == "Glass": return self.glass()
        if name == "Hardware": return self.hardware()
        if name == "SoyBean": return self.soybean()


    def abalone(self):
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
        target_name = 'Rings'
        return (file, columns, target_name)

    def breastcancer(self):
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
        return (file, columns, target_name)

    def forestfires(self):
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
        target_name = 'Area'
        return (file, columns, target_name)

    def glass(self):
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
        return (file, columns, target_name)

    def hardware(self):
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
        return (file, columns, target_name)

    def soybean(self):
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
        target_name = 'Class'
        return (file, columns, target_name)