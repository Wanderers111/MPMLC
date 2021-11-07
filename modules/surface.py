"""
The script is written from the dMaSIF, which is written by
"""

import torch
from pykeops.torch import LazyTensor
from pykeops.torch.cluster import grid_cluster


def soft_distance(x, y, atomtype, smoothness=0.01):
    """

    :param smoothness: float, the smoothing constant for the distance calculation.
    :param x: torch.Tensor, the coordinates for the atoms.
    :param y: torch.Tensor, the coordinates for the neighborhoods for the atoms.
    :param atomtype: torch.LongTensor,
    :return:
    """
    # Return the distances between the atoms and the neighborhood points
    x_i = LazyTensor(x[:, None, :])  # (N * 1 * 3) atom coordinates
    y_i = LazyTensor(y[None, :, :])  # (1 * M * 3) neighborhoods points coordinates.
    D_ij = ((x_i - y_i) ** 2).sum(-1)  # Calculating the distance of the x_i and y_i

    # !TODO: Divide the batches, using block-diagonal sparsity
    # Get the atom radius and normalizing the atom radius.
    # The CSD atom radius for Csp, Csp2, Csp3, H, O, N, P, S, Se, Cl, F, Br. Dalton Transactions 2832-2838
    atom_radius = torch.cuda.FloatTensor([69, 73, 76, 31, 66, 71, 107, 105, 120, 102, 57, 120], device=x.device)
    atom_radius = atom_radius / atom_radius.min()
    # Get the normalized radius of the batch atoms.
    molecule_radius = torch.sum(smoothness * atomtype * atom_radius, dim=1, keepdim=False)
    sigma_molecule = (((-1 * D_ij.sqrt()).exp() * molecule_radius).sum(0) / ((-1 * D_ij.sqrt()).exp()).sum(0))
    molecule_radius_i = LazyTensor(molecule_radius[:, None, None])
    t = -1 * D_ij.sqrt() / molecule_radius_i
    smooth_distance_function = -1 * sigma_molecule * t.exp().sum(0).log()
    return smooth_distance_function


class MoleculeAtomsToPointNormal:
    def __init__(self, atoms, atomtype, theta_distance=1.0, B=500, r=2.05, smoothness=0.1, variance=0.15):
        """
        A class to convert the atoms of a molecule to point normal surface.
        :param atoms: torch.Tensor, (N * 3) tensor represents the coordinates of the molecule.
        :param atomtype: torch.Tensor, the kinds of the atoms input.
        :param theta_distance: float, the variance distance (A) of the normal sampling of the neighborhood points.
        :param B: int, the number of the sampling points.
        :param r: float, the radius of the level set surface.
        :param smoothness: float, the smooth constant for SDF calculation.
        :param variance: float, the distance to process.
        """
        self.atoms = atoms
        self.atomtype = atomtype
        self.theta_distance = theta_distance
        self.B = B
        self.r = r
        self.smoothness = smoothness
        self.variance = variance

    def sampling(self):
        """
        SAMPLING: Sampling B random points in the neighborhood of the atoms of our atoms.
        :return:
        torch.Tensor (N*D, coord_dim) The coordination for the sampling point.
        """
        n_atoms, coord_dim = self.atoms.shape
        z = self.atoms[:, None, :] + 10 * self.theta_distance * torch.randn(n_atoms, self.B, coord_dim, device='cuda')
        z = z.view(-1, coord_dim)
        # Make the tensor's store continuous in the device
        atoms = self.atoms.detach().contiguous()
        z = z.detach().contiguous()
        return atoms, z

    def descend(self, atoms, z, ite=100):
        """
        DESCEND: Update the atom neighborhood vector of the atoms based on the SDF loss
        :param atoms: torch.Tensor, the coordinates of the atoms.
        :param z: torch.Tensor, the coordinates of the neighborhoods atoms.
        :param ite: int, the number of the iterations.
        :return:
        z, torch.Tensor, the updated coordinate vectors for the neighborhoods atoms
        """
        # To set the z's grad True
        if z.is_leaf:
            z.requires_grad = True
        # Update the coordinates of the neighborhoods atoms based on gradient backward.
        # The number of the iterations is 4.
        for ite_i in range(ite):
            smooth_dist = soft_distance(atoms, z, self.atomtype, smoothness=self.smoothness)
            dist_loss = 0.001 * ((smooth_dist - self.r) ** 2).sum()
            grad = torch.autograd.grad(dist_loss, z)[0]
            z = z - 10 * grad
        return z.detach().contiguous()

    def cleaning(self, atoms, z):
        """
        Cleaning: Clean the points based on the threshold < 1.05 A
        :param atoms: torch.Tensor, The coordinates of the atoms.
        :param z: torch.Tensor, The coordinates of the neighborhood point.
        :return:
        torch.Tensor, the final coordinates of the points after being cleaned.
        """
        D_ij_final = ((z[None, :, :] - atoms[:, None, :]) ** 2).sum(-1).sqrt()
        mask = (D_ij_final < self.variance).nonzero()[:, 1]
        mask_i = torch.unique(mask)
        idx = torch.arange(z.shape[0], device=mask_i.device)
        superset = torch.cat([mask_i, idx])
        uniset, count = superset.unique(return_counts=True)
        mask = (count == 1)
        result = uniset.masked_select(mask)
        z_final = z[result]
        return z_final.contiguous()

    @staticmethod
    def sub_sampling(z, solution=0.31):
        """
        SUBSAMPLING: To sub sample the cloud point of the molecular surface.
                    The algorithm is to cluster the points and calculate
        the mean value of each cluster.
        :param solution: float, the solution of the point cloud.
        :param z: torch.FloatTensor, the dimension of the molecule point surface.
        :return:
        The points after cluster.
        """
        grid_class = grid_cluster(z, solution).long()
        grid_class_labels = grid_class.max() + 1
        z_1 = torch.cat((z, torch.ones_like(z[:, 1])[:, None]), 1)
        points = torch.zeros_like(z_1[:grid_class_labels])
        return (points[:, :3] / points[:, 3:]).contiguous()

    def normals(self):
        return
