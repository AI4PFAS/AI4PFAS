import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.feature_selection import VarianceThreshold

from rdkit import Chem
from rdkit.Chem import Descriptors

from helpers import count_cf_bonds, create_morgan_space

def generate_dataset(splitter, name):
    ldtoxdb = pd.read_csv('../data/ldtoxdb-mordred.csv').dropna(axis=1)

    ldtoxdb['rd_mol'] = ldtoxdb.SMI.apply(Chem.MolFromSmiles)
    ldtoxdb['n_cf_bonds'] = ldtoxdb.rd_mol.apply(count_cf_bonds)
    ldtoxdb['mol_wt'] = ldtoxdb.rd_mol.apply(Chem.Descriptors.MolWt)
    ldtoxdb['is_pfas_like'] = ldtoxdb['n_cf_bonds'] >= 2

    mordred = ldtoxdb.columns[5:-5]
    ecfp4096 = np.array(ldtoxdb.rd_mol.apply(create_morgan_space(nbits=4096, r=2)).tolist())
    ecfp2048 = np.array(ldtoxdb.rd_mol.apply(create_morgan_space(nbits=2048, r=1)).tolist())
    ecfp2048r6 = np.array(ldtoxdb.rd_mol.apply(create_morgan_space(nbits=2048, r=6)).tolist())
    graph = np.array(ldtoxdb.rd_mol.apply(mol2graph.mol2torchdata).tolist())


    # for stratified splitting
    bins = pd.cut(ldtoxdb[['NeglogLD50']].to_numpy().reshape(-1), bins=5, labels=False)

    for foldno, (train_idx, test_idx) in enumerate(splitter.split(ldtoxdb, bins)):
        prefix = '../data/preprocessed/%s/fold%d' % (name, foldno)

        train = ldtoxdb.iloc[train_idx]
        test = ldtoxdb.iloc[test_idx]

        pfas_train = train.loc[train.is_pfas_like]
        pfas_test = test.loc[test.is_pfas_like]

        pfas_train_idx = pfas_train.index
        pfas_test_idx = pfas_test.index

        # SMILES
        np.savez_compressed(prefix + '_smiles_test', smiles=test[['SMI']].to_numpy())
        np.savez_compressed(prefix + '_smiles_train', smiles=train[['SMI']].to_numpy())

        np.savez_compressed(prefix + '_smiles_test_pfas_like', smiles=pfas_test[['SMI']].to_numpy())
        np.savez_compressed(prefix + '_smiles_train_pfas_like', smiles=pfas_train[['SMI']].to_numpy())

        # Outputs
        np.savez_compressed(prefix + '_y_test', y=test[['NeglogLD50']].to_numpy())
        np.savez_compressed(prefix + '_y_train', y=train[['NeglogLD50']].to_numpy())

        np.savez_compressed(prefix + '_y_test_pfas_like', y=pfas_test[['NeglogLD50']].to_numpy())
        np.savez_compressed(prefix + '_y_train_pfas_like', y=pfas_train[['NeglogLD50']].to_numpy())

        # Mordred inputs
        col_selector = VarianceThreshold()
        np.savez_compressed(prefix + '_mordred_x_train', x=col_selector.fit_transform(train[mordred]).astype(np.float32))
        np.savez_compressed(prefix + '_mordred_x_test', x=col_selector.transform(test[mordred]).astype(np.float32))

        np.savez_compressed(prefix + '_mordred_x_train_pfas_like', x=col_selector.transform(pfas_train[mordred]).astype(np.float32))
        np.savez_compressed(prefix + '_mordred_x_test_pfas_like', x=col_selector.transform(pfas_test[mordred]).astype(np.float32))

        # We need these for inference later on
        indices = col_selector.get_support(indices=True)
        np.savez_compressed(prefix + '_mordred_x_cols', cols=train[mordred].iloc[[0], indices].columns)

        # ECFP-4096 inputs
        np.savez_compressed(prefix + '_ecfp_4096_x_train', x=col_selector.fit_transform(ecfp4096[train_idx]).astype(np.float32))
        np.savez_compressed(prefix + '_ecfp_4096_x_test', x=col_selector.transform(ecfp4096[test_idx]).astype(np.float32))

        np.savez_compressed(prefix + '_ecfp_4096_x_train_pfas_like', x=col_selector.transform(ecfp4096[pfas_train_idx]).astype(np.float32))
        np.savez_compressed(prefix + '_ecfp_4096_x_test_pfas_like', x=col_selector.transform(ecfp4096[pfas_test_idx]).astype(np.float32))

        # ECFP-2048 inputs
        np.savez_compressed(prefix + '_ecfp_2048_x_train', x=col_selector.fit_transform(ecfp2048[train_idx]).astype(np.float32))
        np.savez_compressed(prefix + '_ecfp_2048_x_test', x=col_selector.transform(ecfp2048[test_idx]).astype(np.float32))

        np.savez_compressed(prefix + '_ecfp_2048_x_train_pfas_like', x=col_selector.transform(ecfp2048[pfas_train_idx]).astype(np.float32))
        np.savez_compressed(prefix + '_ecfp_2048_x_test_pfas_like', x=col_selector.transform(ecfp2048[pfas_test_idx]).astype(np.float32))

        # ECFP-2048 inputs
        np.savez_compressed(prefix + '_ecfp_2048r6_x_train', x=col_selector2.fit_transform(ecfp2048r6[train_idx]).astype(np.float32))
        np.savez_compressed(prefix + '_ecfp_2048r6_x_test', x=col_selector.transform(ecfp2048r6[test_idx]).astype(np.float32))

        np.savez_compressed(prefix + '_ecfp_2048r6_x_train_pfas_like', x=col_selector.transform(ecfp2048r6[pfas_train_idx]).astype(np.float32))
        np.savez_compressed(prefix + '_ecfp_2048r6_x_test_pfas_like', x=col_selector.transform(ecfp2048r6[pfas_test_idx]).astype(np.float32))

        # GP Convienience
        col_selector2 = VarianceThreshold()
        np.savez_compressed(prefix + '_gp_x_train', x=col_selector.fit_transform(train[mordred]).astype(np.float32),
                                                            x2=col_selector2.fit_transform(ecfp4096[train_idx]).astype(np.float32))
        np.savez_compressed(prefix + '_gp_x_test', x=col_selector.transform(test[mordred]).astype(np.float32),
                                                           x2=col_selector2.transform(ecfp4096[test_idx]).astype(np.float32))
        # GCN
        # Graph featurized on fly


def main():
    np.random.seed(9700)
    generate_dataset(splitter=KFold(n_splits=5, shuffle=True), name='random')

    np.random.seed(9700)
    generate_dataset(splitter=StratifiedKFold(n_splits=5, shuffle=True), name='stratified')

if __name__ == '__main__':
    main()