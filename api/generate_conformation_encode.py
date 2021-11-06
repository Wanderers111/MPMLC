import zlib

from rdkit import Chem
from rdkit.Chem.PropertyMol import PropertyMol
import torch

from modules.mol import conformation_generation, get_mol_coordinate, get_mol_type_one_hot

from modules.surface import MoleculeAtomsToPointNormal


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


def decode(block):
    """

    :param block:
    :return:
    """
    string = zlib.decompress(block)
    string = string.decode()
    mols = []
    for string_i in string.split('END\n\n'):
        mols.append(Chem.MolFromMolBlock(string_i + 'END\n\n'))
    return mols


def to_point_cloud(mol, B=500, theta_distance=1.0, r=2.05, smoothness=0.1, variance=0.2, ite=100):
    """

    :param mol: rdkit.mol. The mol object to process the point cloud.
    :param B: int, the number of the sampling points.
    :param theta_distance: float, the variance distance (A) of the normal sampling of the neighborhood points.
    :param r: float, the radius of the level set surface.
    :param smoothness: float, the smooth constant for SDF calculation.
    :param variance: float,
    :param ite: int, The number of the iterations.

    :return:
    """
    conformer = mol.GetConformer()
    atoms = get_mol_coordinate(conformer)
    atomtype = get_mol_type_one_hot(mol)
    atomtype = torch.from_numpy(atomtype).cuda()
    atoms = torch.from_numpy(atoms).cuda()
    point_processer = MoleculeAtomsToPointNormal(atoms=atoms, atomtype=atomtype, B=B, r=r,
                                                 smoothness=smoothness, variance=variance,
                                                 theta_distance=theta_distance)
    atoms, z = point_processer.sampling()
    z = point_processer.descend(atoms, z, ite=ite)
    z = point_processer.cleaning(atoms, z)
    z = point_processer.sub_sampling(z)
    return z
