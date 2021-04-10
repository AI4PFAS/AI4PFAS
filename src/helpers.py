from rdkit import Chem
from rdkit.Chem import AllChem

def create_morgan_space(r = 2, nbits = 128):
    def model(mol):
        return Chem.AllChem.GetMorganFingerprintAsBitVect(mol, radius = r, nBits = nbits)
    return model

def count_cf_bonds(mol):
    abstract_cf = Chem.MolFromSmarts('C~F')
    
    cf_bonds = mol.GetSubstructMatches(abstract_cf)
    return len(cf_bonds)

#@source https://github.com/keras-team/keras/issues/341#issuecomment-539198392
def init_layer(layer):
    #where are the initializers?
    if hasattr(layer, 'cell'):
        init_container = layer.cell
    else:
        init_container = layer

    for key, initializer in init_container.__dict__.items():
        if "initializer" not in key: #is this item an initializer?
                continue #if no, skip it

        # find the corresponding variable, like the kernel or the bias
        if key == 'recurrent_initializer': #special case check
            var = getattr(init_container, 'recurrent_kernel')
        else:
            var = getattr(init_container, key.replace("_initializer", ""))

        var.assign(initializer(var.shape, var.dtype))
        #use the initializer