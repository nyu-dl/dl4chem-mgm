from typing import List

import torch
from guacamol.distribution_matching_generator import DistributionMatchingGenerator


def generate_smiles(params, model, dico, number_samples, sample_temperature=1.0):
    gen_smiles = []
    while len(gen_smiles) < number_samples:
        # run generation of molecules ( regular sampling)
        with torch.no_grad():
            gen, gen_len = model.generate_unconditional(params, max_len=params.max_len, bs=params.batch_size,
                                                        sample_temperature=sample_temperature)
        for b in range(len(gen_len)):
            gen_b = gen[:,b][:gen_len[b]]
            smiles_b = ''.join([dico[symbol.item()] for symbol in gen_b[1:-1]])
            gen_smiles.append(smiles_b)
    return gen_smiles[:number_samples]

class ARMockGenerator(DistributionMatchingGenerator):
    """
    Mock generator that returns pre-defined molecules,
    possibly split in several calls
    """

    def __init__(self, molecules: List[str]) -> None:
        self.molecules = molecules

    def generate(self, number_samples: int) -> List[str]:
        return self.molecules
