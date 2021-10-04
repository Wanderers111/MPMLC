"""
The script is written from the dMaSIF, which is written by
"""

import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from pykeops.torch import LazyTensor

# Generate confomration
conformation_kwargs = {
    'maxAttempts': 100,
    'useExpTorsionAnglePrefs': True,
    'useBasicKnowledge': True
}

# The CSD atom radius for Csp, Csp2, Csp3, H, O, N, P, S, Se, Cl, F, Br. Dalton Transactions 2832-2838
atom_radius = [69, 73, 76, 31, 66, 71, 107, 105, 120, 102, 57, 120]


def conformation_generation(smi, force_field_optimization=True, RmsThresh=0.5, numConfs=100, **kwargs):
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol, numConfs=numConfs, maxAttempts=kwargs['maxAttempts'],
                               useExpTorsionAnglePrefs=kwargs['useExpTorsionAnglePrefs'],
                               useBasicKnowledge=kwargs['useBasicKnowledge'])
    rms_list = []
    AllChem.AlignMolConformers(mol, RMSlist=rms_list)
    if force_field_optimization:
        AllChem.MMFFOptimizeMolecule(mol)
    return mol, rms_list


def soft_distance(x, y, batch_x, batch_y, chemical_features):
    """

    :param x:
    :param y:
    :param batch_x:
    :param batch_y:
    :param chemical_features:
    :return:
    """
    # Return the distances between the atoms and the neighborhood points
    x_i = LazyTensor(x[:, None, :])  # (N * 1 * 3) atom coordinates
    y_i = LazyTensor(y[None, :, :])  # (1 * M * 3) neighborhoods points coordinates.
    D_ij = ((x_i - y_i) ** 2).sum(-1) #

    # Divide the batches, using block-diagonal sparsity


    return D_ij


class MoleculeAtomsToPointNormal:
    def __init__(self, atoms, theta_distance=1.0, B=20, r=1.05, chemical_features=False):
        """
        A class to convert the atoms of a molecule to point normal surface.
        :param atoms: torch.Tensor, (N * 3) tensor represents the coordinates of the molecule.
        :param chemical_features: torch.Tensor, the chemical features of the atoms input.
        :param theta_distance: float, the variance distance (A) of the normal sampling of the neighborhood points.
        :param B: int, the number of the sampling points.
        :param r: float, the radius of the level set surface.
        """
        self.atoms = atoms
        self.chemical_features = chemical_features
        self.theta_distance = theta_distance
        self.B = B
        self.r = r

    def sampling(self):
        """
        Sampling B random points in the neighborhood of the atoms of our atoms.
        :return:
        torch.Tensor (N*D, coord_dim) The coordination for the sampling point.
        """
        # !TODO: verify the theta distance is suitable for the z sampling.
        n_atoms, coord_dim = self.atoms.shape
        z = self.atoms[:, None, :] + 10 * self.theta_distance * torch.randn(n_atoms, self.B, coord_dim)
        z = z.view(-1, coord_dim)
        # Make the tensor's store continuous in the device
        atoms = self.atoms.detach().contigous()
        z = z.detach().contigous()
        return atoms, z

    def descend(self):
        return

    def cleaning(self):
        return

    def sub_sampling(self):
        return

    def normals(self):
        return
