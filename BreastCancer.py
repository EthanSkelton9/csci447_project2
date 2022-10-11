from k_nearest import K_Nearest
from IBL import IBL

class BreastCancer (IBL):
    def __init__(self):
        #list of feature names(excluding class)
        features = [   #column names class at end
            'Id',
            'Clump Thickness',
            'Uniformity of Cell Size' ,
            'Uniformity of Cell Shape',
            'Marginal Adhesion',
            'Single Epithelial Cell Size',
            'Bare Nuclei',
            'Bland Chromatin',
            'Normal Nucleoli',
            'Mitoses'
            ]

        #initiate breast cancer test set
        super().__init__(file = 'breast-cancer-wisconsin.csv', features = features,
                       name = "BreastCancer", classLoc = 'end', replaceValue = '3', classification = True)
