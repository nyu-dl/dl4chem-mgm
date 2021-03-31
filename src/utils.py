# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import getpass
import inspect
import os
import random
import re
import time
from collections import OrderedDict
import copy

import numpy as np
import torch
from guacamol.distribution_learning_benchmark import ValidityBenchmark, UniquenessBenchmark, KLDivBenchmark
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors
from torch import optim
from torch.nn import functional as F

from data.gen_targets import QM9_SYMBOL_LIST

FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}

DUMP_PATH = '/checkpoint/%s/dumped' % getpass.getuser()
DYNAMIC_COEFF = ['lambda_clm', 'lambda_mlm', 'lambda_pc', 'lambda_ae', 'lambda_mt', 'lambda_bt']

DEEPFRI_VOCAB = OrderedDict([
    ('-', 0),
    ('D', 1),
    ('G', 2),
    ('U', 3),
    ('L', 4),
    ('N', 5),
    ('T', 6),
    ('K', 7),
    ('H', 8),
    ('Y', 9),
    ('W', 10),
    ('C', 11),
    ('P', 12),
    ('V', 13),
    ('S', 14),
    ('O', 15),
    ('I', 16),
    ('E', 17),
    ('F', 18),
    ('X', 19),
    ('Q', 20),
    ('A', 21),
    ('B', 22),
    ('Z', 23),
    ('R', 24),
    ('M', 25),
    ('<mask>', 26)])
DEEPFRI_VOCAB_KEYS = list(DEEPFRI_VOCAB.keys())

IUPAC_VOCAB = OrderedDict([
    ("<pad>", 0),
    ("<mask>", 1),
    ("<cls>", 2),
    ("<sep>", 3),
    ("<unk>", 4),
    ("A", 5),
    ("B", 6),
    ("C", 7),
    ("D", 8),
    ("E", 9),
    ("F", 10),
    ("G", 11),
    ("H", 12),
    ("I", 13),
    ("K", 14),
    ("L", 15),
    ("M", 16),
    ("N", 17),
    ("O", 18),
    ("P", 19),
    ("Q", 20),
    ("R", 21),
    ("S", 22),
    ("T", 23),
    ("U", 24),
    ("V", 25),
    ("W", 26),
    ("X", 27),
    ("Y", 28),
    ("Z", 29)])

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def get_index_method():
    if 'post' not in torch.__version__:
        version = [int(i) for i in torch.__version__.split('.')]
    else:
        version = [int(i) for i in torch.__version__.split('.')[:-1]]
    if version[0] >= 1 and version[1] >= 2:
        index_method = 'bool'
    else:
        index_method = 'byte'
    return index_method

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_seed_if(seed):
    """Sets seed only if condition is met. Used to control whether or not seed is set based on user input"""
    if seed < 0:
        seed = int(time.time())
    set_seed(seed)

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")


class AdamInverseSqrtWithWarmup(optim.Adam):
    """
    Decay the LR based on the inverse square root of the update number.
    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (`warmup-init-lr`) until the configured
    learning rate (`lr`). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.
    During warmup:
        lrs = torch.linspace(warmup_init_lr, lr, warmup_updates)
        lr = lrs[update_num]
    After warmup:
        lr = decay_factor / sqrt(update_num)
    where
        decay_factor = lr * sqrt(warmup_updates)
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, warmup_updates=4000, warmup_init_lr=1e-7):
        super().__init__(
            params,
            lr=warmup_init_lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        self.warmup_updates = warmup_updates
        self.warmup_init_lr = warmup_init_lr
        # linearly warmup for the first warmup_updates
        warmup_end_lr = lr
        self.lr_step = (warmup_end_lr - warmup_init_lr) / warmup_updates
        # then, decay prop. to the inverse square root of the update number
        self.decay_factor = warmup_end_lr * warmup_updates ** 0.5
        for param_group in self.param_groups:
            param_group['num_updates'] = 0

    def get_lr_for_step(self, num_updates):
        # update learning rate
        if num_updates < self.warmup_updates:
            return self.warmup_init_lr + num_updates * self.lr_step
        else:
            return self.decay_factor * (num_updates ** -0.5)

    def step(self, closure=None):
        super().step(closure)
        for param_group in self.param_groups:
            param_group['num_updates'] += 1
            param_group['lr'] = self.get_lr_for_step(param_group['num_updates'])


def get_optimizer(parameters, s):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        method = s[:s.find(',')]
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    if method == 'adadelta':
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        optim_fn = optim.Adagrad
    elif method == 'adam':
        optim_fn = optim.Adam
        optim_params['betas'] = (optim_params.get('beta1', 0.9), optim_params.get('beta2', 0.999))
        optim_params.pop('beta1', None)
        optim_params.pop('beta2', None)
    elif method == 'adam_inverse_sqrt':
        optim_fn = AdamInverseSqrtWithWarmup
        optim_params['betas'] = (optim_params.get('beta1', 0.9), optim_params.get('beta2', 0.999))
        optim_params.pop('beta1', None)
        optim_params.pop('beta2', None)
    elif method == 'adamax':
        optim_fn = optim.Adamax
    elif method == 'asgd':
        optim_fn = optim.ASGD
    elif method == 'rmsprop':
        optim_fn = optim.RMSprop
    elif method == 'rprop':
        optim_fn = optim.Rprop
    elif method == 'sgd':
        optim_fn = optim.SGD
        assert 'lr' in optim_params
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    expected_args = inspect.getargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))

    return optim_fn(parameters, **optim_params)


def calculate_acc(correct, total, exception_value):
    try:
        acc = float(correct.cpu().item()) / total * 100
    except:
        acc = exception_value
    return acc

def get_loss_weights(target_ds, ds_scores, equalise, local_cpu):
    if equalise is True:
        weight = torch.from_numpy(len(target_ds) / (np.bincount(target_ds.cpu(),
                                                                minlength=len( ds_scores[0])) + 0.0001)).float()
        if local_cpu is False: weight = weight.cuda()
    else:
        weight = None
    return weight

def normalise_loss(unnormalised_loss, num_ds_components, num_all_ds_components, loss_normalisation_type):
    if loss_normalisation_type == 'by_total':
        loss = unnormalised_loss / num_all_ds_components
    elif loss_normalisation_type == 'by_component':
        loss = unnormalised_loss / num_ds_components
    return loss

def get_loss(target_ds, ds_scores, equalise, loss_normalisation_type, num_components, local_cpu=False, binary=False):
    weight = get_loss_weights(target_ds, ds_scores, equalise, local_cpu)
    if binary is True:
        loss = F.binary_cross_entropy(ds_scores, target_ds, weight=weight, reduction='sum')
    else:
        loss = F.cross_entropy(ds_scores, target_ds, weight=weight, reduction='sum')
    loss = normalise_loss(loss, len(target_ds), num_components, loss_normalisation_type)
    return loss

def get_targetcorresponding_output(output, target_inds, last_dim):
    output = output[target_inds.unsqueeze(-1).expand_as(output)]
    return output.reshape([-1, last_dim])

def reshape_and_get_targetcorresponding_outputs(edge_variables, node_variables,
                                                 node_target_inds_vector, edge_target_coords_matrix):
    reshaped_node_variables, reshaped_edge_variables = [], []
    for nv in node_variables:
        reshaped_node_variables.append(get_targetcorresponding_output(nv, node_target_inds_vector, nv.shape[-1]))
    for ev in edge_variables:
        ev = ev.reshape(*ev.shape[:-2], ev.shape[-2] * ev.shape[-1])
        reshaped_edge_variables.append(
            get_targetcorresponding_output(ev, edge_target_coords_matrix, ev.shape[-1]))
    return reshaped_edge_variables, reshaped_node_variables

def get_only_target_info(ds_scores, original_ds_inds, ds_target_inds_vector, ds_num_classes, ds_target_types):
    """Return information on only those data structures that are predicted"""
    ds_scores = get_targetcorresponding_output(ds_scores, ds_target_inds_vector, ds_num_classes)
    target_ds = original_ds_inds[ds_target_inds_vector]
    ds_target_types = ds_target_types[ds_target_inds_vector]
    return ds_scores, target_ds, ds_target_types


def get_ds_stats(ds_scores, target_ds, ds_target_types):
    ds_preds = torch.argmax(F.softmax(ds_scores, -1), dim=-1)
    ds_correct = (ds_preds == target_ds).sum()
    mask_ds_correct = torch.mul((ds_preds == target_ds), (ds_target_types == 1)).sum()
    replace_ds_correct = torch.mul((ds_preds == target_ds), (ds_target_types == 2)).sum()
    recon_ds_correct = torch.mul((ds_preds == target_ds), (ds_target_types == 3)).sum()
    ds_acc = float(ds_correct.cpu().item()) / float(ds_scores.shape[0]) * 100
    mask_ds_acc = calculate_acc(mask_ds_correct, float((ds_target_types == 1).sum().item()), np.nan)
    replace_ds_acc = calculate_acc(replace_ds_correct, float((ds_target_types == 2).sum().item()), np.nan)
    recon_ds_acc = calculate_acc(recon_ds_correct, float((ds_target_types == 3).sum().item()), np.nan)
    return ds_acc, mask_ds_acc, replace_ds_acc, recon_ds_acc

def get_majority_and_minority_stats(ds_preds, target_ds, majority_index):
    nonmajority_ds_correct = torch.mul((ds_preds == target_ds), (target_ds != majority_index)).sum()
    majority_ds_correct = torch.mul((ds_preds == target_ds), (target_ds == majority_index)).sum()
    nonmajority_ds_acc = calculate_acc(nonmajority_ds_correct, float((target_ds != majority_index).sum().item()), np.nan)
    majority_ds_acc = calculate_acc(majority_ds_correct, float((target_ds == majority_index).sum().item()), np.nan)
    return majority_ds_correct, nonmajority_ds_correct, majority_ds_acc, nonmajority_ds_acc

def check_validity(nodes, edges, is_valid_list=[], is_connected_list=[]):
    for b in range(nodes.shape[0]):
        mol = graph_to_mol(nodes[b], edges[b])
        if mol is None:
            is_valid_list.append(0)
        else:
            if '.' in Chem.MolToSmiles(mol):
                is_connected_list.append(0)
            else:
                is_connected_list.append(1)
            is_valid_list.append(1)
    return is_valid_list, is_connected_list


def write_tensorboard(writer, data_split, results_dict, iterations):
    for name, value in results_dict.items():
        if type(value) != np.nan:
            writer.add_scalar('{}/{}'.format(data_split, name), value, iterations)


def get_result_distributions(preds, targets, target_types, desired_type, bin_length=4):
    target_coords = target_types == desired_type
    preds, targets = preds[target_coords], targets[target_coords]
    correct_dist = np.bincount(targets[preds == targets].cpu().numpy(), minlength=bin_length)
    incorrect_pred_dist = np.bincount(preds[preds != targets].cpu().numpy(), minlength=bin_length)
    incorrect_true_dist = np.bincount(targets[preds != targets].cpu().numpy(), minlength=bin_length)
    return np.vstack((correct_dist, incorrect_pred_dist, incorrect_true_dist))


def get_all_result_distributions(preds, targets, target_types, desired_types, bin_length=5):
    dists = []
    for t in desired_types:
        dists.append(get_result_distributions(preds, targets, target_types, t, bin_length))
    return np.stack(dists)


def update_ds_stats(ds_scores, target_ds, ds_target_types, ds_correct, mask_ds_correct,
                    replace_ds_correct, recon_ds_correct, total_mask_ds, total_replace_ds, total_recon_ds, total_ds):
    ds_preds = torch.argmax(F.softmax(ds_scores, -1), dim=-1)
    ds_correct += (ds_preds == target_ds).sum()
    mask_ds_correct += torch.mul((ds_preds == target_ds), (ds_target_types == 1)).sum()
    replace_ds_correct += torch.mul((ds_preds == target_ds), (ds_target_types == 2)).sum()
    recon_ds_correct += torch.mul((ds_preds == target_ds), (ds_target_types == 3)).sum()
    total_mask_ds += (ds_target_types == 1).sum()
    total_replace_ds += (ds_target_types == 2).sum()
    total_recon_ds += (ds_target_types == 3).sum()
    total_ds += ds_preds.numel()
    return ds_correct, mask_ds_correct, replace_ds_correct, recon_ds_correct, total_mask_ds, \
           total_replace_ds, total_recon_ds, total_ds


def accuracies_from_totals(result_pairs):
    accuracies = []
    for correct, total in result_pairs.items():
        accuracies.append(calculate_acc(correct, float(total), np.nan))
    return accuracies

def lr_decay_multiplier(iteration, warm_up_iters, decay_start_iter, lr_decay_amount, lr_decay_frac, lr_decay_interval,
                        min_lr, lr):
    if iteration < warm_up_iters:
        return iteration/warm_up_iters
    elif iteration - decay_start_iter < 0:
        return 1.0
    else:
        if lr_decay_amount == 0:
            return max(lr_decay_frac ** ((iteration - decay_start_iter) // lr_decay_interval),
                min_lr/lr)
        else:
            return max(1 - (lr_decay_amount * ((iteration - decay_start_iter) // lr_decay_interval))/lr,
                min_lr/lr)

def save_checkpoints(total_iter, avg_val_loss, best_loss, model_state_dict, opt_state_dict, exp_path, logger,
                no_save, save_all):
    if avg_val_loss <= best_loss:
        logger.info('best model')
        best_loss = avg_val_loss
        torch.save(model_state_dict, os.path.join(exp_path, 'best_model'))
        torch.save(opt_state_dict, os.path.join(exp_path, 'best_opt_sd'))

    if no_save is False:
        if save_all is True:
            torch.save(model_state_dict, os.path.join(exp_path, 'model_{}'.format(total_iter)))
        torch.save(model_state_dict, os.path.join(exp_path, 'latest_model'))
        torch.save(opt_state_dict, os.path.join(exp_path, 'latest_opt_sd'))

    return best_loss

def calculate_gen_benchmarks(generator, gen_num_samples, training_smiles, logger):
    try:
        benchmark = ValidityBenchmark(number_samples=gen_num_samples)
        validity_score = benchmark.assess_model(generator).score
    except:
        validity_score = -1
    try:
        benchmark = UniquenessBenchmark(number_samples=gen_num_samples)
        uniqueness_score = benchmark.assess_model(generator).score
    except:
        uniqueness_score = -1
    try:
        benchmark = KLDivBenchmark(number_samples=gen_num_samples, training_set=training_smiles)
        kldiv_score = benchmark.assess_model(generator).score
    except:
        kldiv_score = -1
    logger.info(
        'Validity Score={}, Uniqueness Score={}, KlDiv Score={}'.format(validity_score, uniqueness_score,
                                                                        kldiv_score))

def graph_to_mol(init_node_properties, init_edge_properties, min_charge=-1,
                 node_target_inds=None, edge_target_coords_mat=None, return_pos=False,
                 symbol_list=QM9_SYMBOL_LIST):
    mol = Chem.EditableMol(Chem.Mol())
    node_highlight_pos, edge_highlight_pos = [], []
    for i, node_ind in enumerate(init_node_properties['node_type']):
        if not 0 <= node_ind < len(symbol_list): continue
        atom = Chem.Atom(symbol_list[node_ind])
        if init_node_properties['charge'] is not None:
            atom.SetFormalCharge(init_node_properties['charge'][i].item() - abs(min_charge))
        if init_node_properties['chirality'] is not None:
            chiral_tag = Chem.ChiralType.values[int(init_node_properties['chirality'][i])]
            atom.SetChiralTag(chiral_tag)
        mol.AddAtom(atom)
        if node_target_inds is not None and node_target_inds[i] == 1:
            node_highlight_pos.append(i)
    adj_mat_coords = list(zip(*np.triu_indices(len(init_node_properties['node_type']), k=1)))
    edge_id = 0
    for i, j in adj_mat_coords:
        edge = int(init_edge_properties['edge_type'][i, j])
        if edge in BOND_TYPES.keys():
            mol.AddBond(int(i), int(j), BOND_TYPES[edge])
            if edge_target_coords_mat is not None and int(edge_target_coords_mat[i, j]) == 1:
                edge_highlight_pos.append(edge_id)
            edge_id += 1
    mol = mol.GetMol()
    if init_node_properties['chirality'] is not None:
        try:
            mol.UpdatePropertyCache()
            for atom in mol.GetAtoms():
                if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED: atom.SetNumExplicitHs(atom.GetTotalNumHs())
        except:
            pass
    if return_pos is True:
        return mol, node_highlight_pos, edge_highlight_pos
    else:
        return mol

def calculate_kldiv(mean_q, logvar_q, mean_p=None, logvar_p=None):
    """
    Calculate KL divergence KL[q||p] between two Gaussians with diagonal covariance.
    If p is not given, set it to have mean 0 and covariance I.
    """
    if mean_p is None and logvar_p is None:
        mean_p = torch.zeros_like(mean_q)
        logvar_p = torch.zeros_like(logvar_q)
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    return 0.5 * (logvar_p - logvar_q - 1 + var_q/var_p + (mean_q-mean_p)**2/var_p).sum()

BOND_TYPES = {1: Chem.BondType.SINGLE,
             2: Chem.BondType.DOUBLE,
             3: Chem.BondType.TRIPLE,
             4: Chem.BondType.AROMATIC}

def filter_top_k(logits, k, filter_value=-99999):
    if 0 < k < logits.shape[-1]:
        indices_to_remove = logits < torch.topk(logits, k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    return logits

def calculate_graph_properties(mols, property_names):
    graph_property_mapper = {'molwt': Descriptors.MolWt, 'logp': Crippen.MolLogP}
    graph_properties = {}
    for name in property_names:
        graph_properties[name] = [graph_property_mapper[name](m) for m in mols]
    return graph_properties

def dct_to_cuda_inplace(dct):
    for key in dct.keys():
        dct[key] = dct[key].cuda()

def copy_graph_remove_data(input_graph):
    output_graph = copy.deepcopy(input_graph)
    for key in input_graph.ndata.keys(): del output_graph.ndata[key]
    for key in input_graph.edata.keys(): del output_graph.edata[key]
    return output_graph