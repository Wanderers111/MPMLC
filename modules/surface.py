"""
The script is written from the dMaSIF, which is written by
"""

import torch
from pykeops.torch import LazyTensor


def soft_distance(x, y, atomtype, smoothness=0.01, chemical_features=None):
    """

    :param smoothness:
    :param x:
    :param y:
    :param atomtype:
    :param chemical_features:
    :return:
    """
    # Return the distances between the atoms and the neighborhood points
    x_i = LazyTensor(x[:, None, :])  # (N * 1 * 3) atom coordinates
    y_i = LazyTensor(y[None, :, :])  # (1 * M * 3) neighborhoods points coordinates.
    D_ij = ((x_i - y_i) ** 2).sum(-1)  # Calculating the distance of the x_i and y_i

    # !TODO: Divide the batches, using block-diagonal sparsity
    if chemical_features:
        # Get the atom radius and normalizing the atom radius.
        # The CSD atom radius for Csp, Csp2, Csp3, H, O, N, P, S, Se, Cl, F, Br. Dalton Transactions 2832-2838
        atom_radius = torch.FloatTensor([69, 73, 76, 31, 66, 71, 107, 105, 120, 102, 57, 120], device=x.device)
        atom_radius = atom_radius / atom_radius.min()
        # Get the normalized radius of the batch atoms.
        molecule_radius = torch.sum(smoothness * atomtype * atom_radius, dim=1, keepdim=False)
        molecule_radius_i = molecule_radius[:, None, None]
        sigma_molecule = (-1*D_ij).exp() * molecule_radius_i / (-1*D_ij).exp()
        smooth_distance_function = -1 * sigma_molecule * (-1*D_ij/molecule_radius_i).exp().sum(-1).log()
    return D_ij


class MoleculeAtomsToPointNormal:
    def __init__(self, atoms, theta_distance=1.0, B=20, r=1.05, chemical_features=None):
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
        atoms = self.atoms.detach().contiguous()
        z = z.detach().contiguous()
        return atoms, z

    def descend(self):
        return

    def cleaning(self):
        return

    def sub_sampling(self):
        return

    def normals(self):
        return
