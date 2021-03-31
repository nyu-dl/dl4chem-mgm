import argparse
import os
import time

import torch
# guacamol stuff
from guacamol.assess_distribution_learning import assess_distribution_learning
from guacamol.distribution_matching_generator import DistributionMatchingGenerator

from ar_mock_generator import generate_smiles
from src.data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
from src.model.transformer import TransformerModel
from src.utils import set_seed, AttrDict


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Generate molecular smiles with transformer")

    # main parameters
    parser.add_argument("--dump_path", type=str, default='dumped/', help="Experiment dump path")
    parser.add_argument("--model_path", type=str, help="Model path")
    parser.add_argument("--dist_file", type=str, help="Distribution file, file pointing to all available molecules")
    parser.add_argument("--suite", type=str, default="v2",
                        help="Suite of guacamol")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed")

    parser.add_argument("--batch_size", type=int, default=100,
                        help="Number of sentences per batch")
    parser.add_argument("--bptt", type=int, default=1024,
                        help="Sequence length")
    parser.add_argument("--max_len", type=int, default=300,
                        help="Maximum length of sentences (after splitting into chars)")
    parser.add_argument('--local_cpu', action='store_true')
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--sample_temperature', type=float, default=1.0)
    parser.add_argument('--output_dir', default='')
    return parser

def reload_ar_checkpoint(path):
    """ Reload autoregressive params, dictionary, model from a given path """
    # Load dictionary/model/datasets first
    reloaded = torch.load(path)
    params = AttrDict(reloaded['params'])

    # build dictionary / update parameters
    dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])
    params.n_words = len(dico)
    params.n_langs = 1
    params.bos_index = dico.index(BOS_WORD)
    params.eos_index = dico.index(EOS_WORD)
    params.pad_index = dico.index(PAD_WORD)
    params.unk_index = dico.index(UNK_WORD)
    params.mask_index = dico.index(MASK_WORD)

    # build Transformer model
    model = TransformerModel(params, is_encoder=False, with_output=True)
    model.load_state_dict(reloaded['model'])
    return params, dico, model

class SmilesTransformerGenerator(DistributionMatchingGenerator):
    """
    Wraps SmilesTransformer in a class satisfying the DistributionMatchingGenerator interface.
    """

    def __init__(self, params, dico, model, sample_temperature):
        self.params = params
        self.model = model
        self.dico = dico
        self.sample_temperature = sample_temperature

    def generate(self, number_samples):
        return generate_smiles(self.params, self.model, self.dico, number_samples, self.sample_temperature)


def main(params):
    # setup random seeds
    set_seed(params.seed)
    params.ar = True

    if not os.path.isdir(params.output_dir): os.makedirs(params.output_dir)
    print("Loading the model from {0}".format(params.model_path))
    # load everything from checkpoint
    model_params, dico, model = reload_ar_checkpoint(params.model_path)
    if params.local_cpu is False:
        model = model.cuda()
    # evaluate distributional results
    generator = SmilesTransformerGenerator(params, dico, model, params.sample_temperature)
    json_file_path = os.path.join(params.output_dir, 'distribution_learning_results.json')
    smiles_output_path = os.path.join(params.output_dir, 'generated_smiles.txt')
    print("Starting distributional evaluation")
    t1 = time.time()
    if params.evaluate is True:
        assess_distribution_learning(generator,
                                 chembl_training_file=params.dist_file,
                                 json_output_file=json_file_path,
                                 benchmark_version=params.suite)
    else:
        smiles_list = generator.generate(params.num_samples)
        with open(smiles_output_path, 'w') as f:
            for smiles in smiles_list:
                f.write(smiles + '\n')
    t2 = time.time()
    print ("Total time taken {}".format(t2-t1))

if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # run experiment
    main(params)
