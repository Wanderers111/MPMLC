from multiprocessing import Pool
from rdkit import Chem
from rdkit.Chem.PropertyMol import PropertyMol
from modules.mol import conformation_generation, get_mol_coordinate, get_mol_type_one_hot
import zlib
import pandas as pd
import pickle


def generate_and_encode(smi):
    """
    Generate the conformation of a specific smiles.
    :param smi:
    :return:
    """
    mol, idxs = conformation_generation(smi, RmsThresh=1)
    mol_blocks_list = []
    for idx in idxs:
        mol_clone = PropertyMol(mol)
        conformer = mol.GetConformer(idx)
        mol_clone.AddConformer(conformer)
        mol_clone.SetProp('_Name', smi)
        mol_blocks_list.append(Chem.MolToMolBlock(mol_clone, ))
    s = '\n'.join(mol_blocks_list)
    s = s.encode()
    zlib_s = zlib.compress(s)
    del mol
    del mol_blocks_list
    return zlib_s


if __name__ == '__main__':
    table = pd.read_csv('./zinc15_druglike_clean_canonical_max60.smi', header=None)
    smis = table[0].loc[:100000].values.tolist()
    processing = Pool(10, maxtasksperchild=1000)
    zlibs_list = []
    for smi_i in smis:
        zlibs_list.append(processing.apply_async(generate_and_encode, args=(smi_i,)))
    processing.close()
    processing.join()
    zlibs_final_list = [zlib_i.get() for zlib_i in zlibs_list]
    with open('dataset.mdb', 'wb') as f:
        pickle.dump(zlibs_final_list, f)
