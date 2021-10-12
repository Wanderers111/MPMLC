import zlib

from rdkit import Chem
from rdkit.Chem.PropertyMol import PropertyMol

from modules.mol import conformation_generation


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
