import pickle
from argparse import ArgumentParser

import numpy as np
import scipy.sparse
from rdkit import Chem
from tqdm import tqdm

parser = ArgumentParser()

parser.add_argument('--data-path', default='data/QM9/QM9_smiles.txt')
parser.add_argument('--save-path', default='data/QM9/QM9_processed.p')
parser.add_argument('--dataset-type', default='QM9')

QM9_SYMBOL_LIST = np.array(['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Ge', 'As', 'Se', 'Br', 'Te', 'I'])
CHEMBL_SYMBOL_LIST = np.array(['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Se', 'Br', 'I'])

def get_symbol_list(dataset_type):
    if dataset_type == 'QM9':
        symbol_list = QM9_SYMBOL_LIST
    elif dataset_type == 'ChEMBL':
        symbol_list = CHEMBL_SYMBOL_LIST
    return symbol_list

def load_data(data_path):
    mols = []
    with open(data_path) as f:
        line = f.readline()
        while line:
            line = line.strip()
            mol = Chem.MolFromSmiles(line)
            mols.append(mol)
            line = f.readline()
    return mols

def mols_to_graphs(mols, symbol_list, print_progress=False):
    mol_infos = []
    if print_progress is True: print(len(mols), flush=True)
    for mol in tqdm(mols):
        atoms = mol.GetAtoms()
        bonds = Chem.GetAdjacencyMatrix(mol, useBO=True)
        bonds[np.where(bonds == 1.5)] = 4
        bonds = scipy.sparse.csr_matrix(bonds)
        atomic_inds, num_hs, charge, is_in_ring, is_aromatic, chirality = [], [], [], [], [], []
        for atom in atoms:
            atomic_symbol = atom.GetSymbol()
            atomic_ind = np.where(atomic_symbol == symbol_list)[0][0]
            atomic_inds.append(atomic_ind)
            num_hs.append(atom.GetTotalNumHs())
            charge.append(atom.GetFormalCharge())
            is_in_ring.append(atom.IsInRing())
            is_aromatic.append(atom.GetIsAromatic())
            chirality.append(int(atom.GetChiralTag()))
        mol_infos.append((np.array(atomic_inds), bonds, np.array(num_hs), np.array(charge), np.array(is_in_ring),
                          np.array(is_aromatic), np.array(chirality)))
    return mol_infos

def main(data_path, save_path, dataset_type):
    symbol_list = get_symbol_list(dataset_type)
    print ("loading data")
    mols = load_data(data_path)
    print ("done loading data")
    mol_infos = mols_to_graphs(mols, symbol_list, print_progress=True)
    with open(save_path, 'wb') as f:
        pickle.dump(mol_infos, f)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args.data_path, args.save_path, args.dataset_type)
