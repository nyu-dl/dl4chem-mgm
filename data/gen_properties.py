import pickle
from argparse import ArgumentParser

from rdkit.Chem import Crippen, Descriptors
from tqdm import tqdm
from data.gen_targets import load_QM9, load_ChEMBL

parser = ArgumentParser()

parser.add_argument('--data-path', default='data/QM9_molset_all.p')
parser.add_argument('--save-path', default='data/QM9_properties.p')
parser.add_argument('--dataset-type', default='QM9')

def get_properties(mols):
    properties = []
    for mol in tqdm(mols):
        molwt = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        properties.append((molwt, logp))
    return properties

def main(data_path, save_path, dataset_type):
    print ("loading data")
    if dataset_type == 'QM9':
        mols, smiles = load_QM9(data_path)
    elif dataset_type == 'ChEMBL':
        mols, smiles = load_ChEMBL(data_path)
    print ("done loading data")
    graph_properties = get_properties(mols)
    with open(save_path, 'wb') as f:
        pickle.dump(graph_properties, f)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args.data_path, args.save_path, args.dataset_type)