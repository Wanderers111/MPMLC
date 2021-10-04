import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# Generate conformation
conformation_kwargs = {
    'maxAttempts': 100,
    'useExpTorsionAnglePrefs': True,
    'useBasicKnowledge': True,
}
atom_id = [6, 1, 8, 7, 15, 16, 34, 17, 9, 35]


def conformation_generation(smi, force_field_optimization=True, RmsThresh=0.5, numConfs=100):
    """
    Generate the specific number of the conformations.
    :param smi: str, The SMILES of the molecule.
    :param force_field_optimization: bool, Whether the molecule should be optimized.
    :param RmsThresh: float, the thresh of the RMSD to remove the duplicate conformers.
    :param numConfs: int, the maximum number of the conformations should be generated.
    :return:
    mol: Rdkit.mol object.
    rms_list: List, The list containing rmsds.
    """
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol, numConfs=numConfs, maxAttempts=conformation_kwargs['maxAttempts'],
                               useExpTorsionAnglePrefs=conformation_kwargs['useExpTorsionAnglePrefs'],
                               useBasicKnowledge=conformation_kwargs['useBasicKnowledge'])
    rms_list = []
    AllChem.AlignMolConformers(mol, RMSlist=rms_list)
    if force_field_optimization:
        AllChem.MMFFOptimizeMolecule(mol)
    conformers_list = list(mol.GetConformers())
    mol_clone = Chem.Mol(mol)
    mol_clone.RemoveAllConformers()
    for idx, conformer in enumerate(conformers_list):
        for i in range(idx + 1, len(conformers_list)):
            distance = np.sqrt(
                ((conformer.GetPositions() - conformers_list[i].GetPositions()) ** 2).sum() / mol.GetNumAtoms()
            )
            if distance < 0.5:
                break
        else:
            mol_clone.AddConformer(conformer)
    return mol_clone


def get_mol_type_one_hot(mol):
    """
    Generate a matrix for the molecule's atom type matrix.
    :param mol: A Rdkit.Mol Object
    :return:
    np.array: The matrix containing the one-hot vector of the atom kind.
    """
    atom_matrix = []
    for atom in mol.GetAtoms():
        atom_id_vector = np.zeros([len(atom_id) + 2])
        if atom.GetAtomicNum() == 6:
            if str(atom.GetHybridization()) == 'SP3':
                atom_id_vector[0] = 1
            elif str(atom.GetHybridization()) == 'SP2':
                atom_id_vector[1] = 1
            elif str(atom.GetHybridization()) == 'SP':
                atom_id_vector[2] = 1
        else:
            atom_id_vector[atom_id.index(atom.GetAtomicNum()) + 2] = 1
        atom_matrix.append(atom_id_vector)
    return atom_matrix


def get_mol_coordinate(conformer):
    """
    Get the coordination vectors of the molecule.
    :param conformer: Rdkit.Conformer.
    :return:
    np.array, The numpy array of the coordinates for the conformer.
    """
    return conformer.GetPositions()
