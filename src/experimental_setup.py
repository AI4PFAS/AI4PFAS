import pandas as pd
import numpy as np

from src import dataset

from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler

path_prefix = ''
scaler = StandardScaler()

class LD50UnitConverter():
    def convert_to_mgkg(self, neglogld50s, smiles):

        for neglogld50, smile in zip(neglogld50s, smiles):
            molwt = Descriptors.MolWt(Chem.MolFromSmiles(smile[0]))
            yield (10**(-1*neglogld50[0]))*1000*molwt


    def convert_to_epa(self, neglogld50s, smiles):
        mgkg = list(self.convert_to_mgkg(neglogld50s=neglogld50s, smiles=smiles))

        return pd.cut(mgkg, labels=(0,1,2,3), bins=(-np.inf,50,500,5000, np.inf))

class CrossValidator():
    def __init__(self, splits = 5, sampling_type = 'random'):
        self.sampling_stratified = sampling_type == 'stratified'
        self.splits = splits

    def get_folds(self, encoding, subset = None):
       for fold in range(self.splits):
            yield dataset.load_dataset(encoding, fold, stratified=self.sampling_stratified, subset=subset,
                path_prefix = path_prefix)