import os
import pickle as pkl

import numpy as np
import sparse
from rdkit import Chem
from rdkit.Chem import Descriptors

data = os.environ.get('DATAPATH')
if data is None:
    data = '/Users/showmaker/Project/MPNN_Generation/data'
use_AROMATIC = True


def to_onehot(val, cat):
    vec = np.zeros(len(cat))
    for i, c in enumerate(cat):
        if val == c:
            vec[i] = 1
    if np.sum(vec) == 0:
        print('* exception: missing category', val)
    assert np.sum(vec) == 1

    return vec


def atomFeatures(a):
    v1 = to_onehot(a.GetFormalCharge(), [-1, 1, 0])[:2]
    v2 = to_onehot(a.GetNumExplicitHs(), [1, 2, 3, 0])[:3]
    v3 = to_onehot(a.GetSymbol(), atom_list)
    return np.concatenate([v1, v2, v3], axis=0)


def bondFeatures(bond):
    e1 = to_onehot(str(bond.GetBondType()), ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'])[:dim_edge]
    return np.array(e1)


def define_the_dimension(atom_list, use_aromatic):
    atom_dim = 5 + len(atom_list)
    bond_dim = 3
    if use_aromatic:
        bond_dim += 1
    return atom_dim, bond_dim


def build_molecule_graph(i, smi):

    return


if data == 'QM9':
    data_size = 100000
    n_max = 9
    atom_list = ['C', 'N', 'O', 'F']

elif data == 'ZINC':
    data_size = 100000
    n_max = 38
    atom_list = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']

# define the node dimension
dim_node = 5 + len(atom_list)

# define the dim edge
dim_edge = 3
if use_AROMATIC:
    dim_edge = dim_edge + 1

smisuppl = pkl.load(open('./' + data + '_smi.pkl', 'rb'))

DV = []
DE = []
DY = []
Dsmi = []
for i, smi in enumerate(smisuppl):
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=False))
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    if not use_AROMATIC:
        Chem.Kekulize(mol)
    n_atom = mol.GetNumHeavyAtoms()

    # node DV
    node = np.zeros((n_max, dim_node), dtype=bool)
    for j in range(n_atom):
        atom = mol.GetAtomWithIdx(j)
        node[j, :] = atomFeatures(atom)

    # edge DE
    edge = np.zeros((n_max, n_max, dim_edge), dtype=bool)
    for j in range(n_atom - 1):
        for k in range(j + 1, n_atom):
            bond = mol.GetBondBetweenAtoms(j, k)
            if bond is not None:
                edge[j, k, :] = bondFeatures(bond)
                edge[k, j, :] = edge[j, k, :]

    # property DY
    property = [Descriptors.ExactMolWt(mol), Descriptors.MolLogP(mol)]

    # append
    DV.append(node)
    DE.append(edge)
    DY.append(property)

    if use_AROMATIC:
        Dsmi.append(Chem.MolToSmiles(mol))
    else:
        Dsmi.append(Chem.MolToSmiles(mol, kekuleSmiles=True))

    if i % 1000 == 0:
        print(i, flush=True)

    if len(DV) == data_size: break

# np array
DV = np.asarray(DV, dtype=bool)
DE = np.asarray(DE, dtype=bool)
DY = np.asarray(DY)
Dsmi = np.asarray(Dsmi)

# compression
DV = sparse.COO.from_numpy(DV)
DE = sparse.COO.from_numpy(DE)

# save
with open(data + '_graph.pkl', 'wb') as fw:
    pkl.dump([DV, DE, DY, Dsmi], fw)
