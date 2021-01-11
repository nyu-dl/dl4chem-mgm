import os
import pickle
from logging import getLogger

import numpy as np
import scipy.sparse as sparse
import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data import Dataset as TorchDataset

from src.utils import set_seed_if
from .dataset import Dataset
from .dictionary import BOS_INDEX, EOS_INDEX, SEP_INDEX, PAD_INDEX, EXTRA_INDEX
from .dictionary import BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD

logger = getLogger()


def process_binarized(data, params):
    """
    Process a binarized dataset and log main statistics.
    """
    dico = data['dico']
    assert ((data['sentences'].dtype == np.uint16) and (len(dico) < 1 << 16) or
            (data['sentences'].dtype == np.int32) and (1 << 16 <= len(dico) < 1 << 31))
    logger.info("%i words (%i unique) in %i sentences. %i unknown words (%i unique) covering %.2f%% of the data." % (
        len(data['sentences']) - len(data['positions']),
        len(dico), len(data['positions']),
        sum(data['unk_words'].values()), len(data['unk_words']),
        100. * sum(data['unk_words'].values()) / (len(data['sentences']) - len(data['positions']))
    ))
    if params.max_vocab != -1:
        assert params.max_vocab > 0
        logger.info("Selecting %i most frequent words ..." % params.max_vocab)
        dico.max_vocab(params.max_vocab)
        data['sentences'][data['sentences'] >= params.max_vocab] = dico.index(UNK_WORD)
        unk_count = (data['sentences'] == dico.index(UNK_WORD)).sum()
        logger.info("Now %i unknown words covering %.2f%% of the data."
                    % (unk_count, 100. * unk_count / (len(data['sentences']) - len(data['positions']))))
    if params.min_count > 0:
        logger.info("Selecting words with >= %i occurrences ..." % params.min_count)
        dico.min_count(params.min_count)
        data['sentences'][data['sentences'] >= len(dico)] = dico.index(UNK_WORD)
        unk_count = (data['sentences'] == dico.index(UNK_WORD)).sum()
        logger.info("Now %i unknown words covering %.2f%% of the data."
                    % (unk_count, 100. * unk_count / (len(data['sentences']) - len(data['positions']))))
    if (data['sentences'].dtype == np.int32) and (len(dico) < 1 << 16):
        logger.info("Less than 65536 words. Moving data from int32 to uint16 ...")
        data['sentences'] = data['sentences'].astype(np.uint16)
    return data


def load_binarized(path, params):
    """
    Load a binarized dataset.
    """
    assert path.endswith('.pth')
    if params.debug_train:
        path = path.replace('train', 'valid')
    if getattr(params, 'multi_gpu', False):
        split_path = '%s.%i.pth' % (path[:-4], params.local_rank)
        if os.path.isfile(split_path):
            assert params.split_data is False
            path = split_path
    assert os.path.isfile(path), path
    logger.info("Loading data from %s ..." % path)
    data = torch.load(path)
    data = process_binarized(data, params)
    return data


def set_dico_parameters(params, data, dico):
    """
    Update dictionary parameters.
    """
    if 'dico' in data:
        assert data['dico'] == dico
    else:
        data['dico'] = dico

    n_words = len(dico)
    bos_index = dico.index(BOS_WORD)
    eos_index = dico.index(EOS_WORD)
    pad_index = dico.index(PAD_WORD)
    unk_index = dico.index(UNK_WORD)
    mask_index = dico.index(MASK_WORD)
    if hasattr(params, 'bos_index'):
        assert params.n_words == n_words
        assert params.bos_index == bos_index
        assert params.eos_index == eos_index
        assert params.pad_index == pad_index
        assert params.unk_index == unk_index
        assert params.mask_index == mask_index
    else:
        params.n_words = n_words
        params.bos_index = bos_index
        params.eos_index = eos_index
        params.pad_index = pad_index
        params.unk_index = unk_index
        params.mask_index = mask_index

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


class QM9DatasetAR(TorchDataset):
    """Autoregressive dataset loader"""
    def __init__(self, graph_infos, params, train=True):
        # set parameters
        self.max_nodes = params.max_nodes
        self.num_node_types = params.num_node_types
        self.num_edge_types = params.num_edge_types

        if params.targets is True:
            self.mol_nodeinds, self.adj_mats, self.targets = zip(*graph_infos)
        else:
            self.mol_nodeinds, self.adj_mats = zip(*graph_infos)
            self.targets = None

        self.train = train

    def __getitem__(self, index):
        # node matrix
        unpadded_node_inds = self.mol_nodeinds[index]
        num_nodes = len(unpadded_node_inds)
        # edge matrix
        unpadded_adj_mat = self.adj_mats[index].todense().astype(int)
        assert (check_symmetric(unpadded_adj_mat.astype(int))) # make sure bond matrix is symmetric
        assert (unpadded_adj_mat.shape[0] == num_nodes and unpadded_adj_mat.shape[1] == num_nodes)

        # represent molecule in a list
        # [bos_index, node1, node2, ..., sep_index, edge12, edge13, ..., eos_index]
        molecule = [BOS_INDEX]

        # add nodes to molecule list
        # make sure node indices start with >= EXTRA_INDEX
        molecule = molecule + list(unpadded_node_inds + EXTRA_INDEX)

        # first raster scan *upper triangular matrix* of edges
        triu_rows_inds, triu_columns_inds = np.triu_indices(num_nodes, 1)
        adj_mat_list = unpadded_adj_mat[triu_rows_inds,triu_columns_inds]
        adj_mat_list = np.array(adj_mat_list)[0] + EXTRA_INDEX + self.num_node_types
        adj_mat_list = adj_mat_list.tolist()

        # add start of edge index first
        molecule = molecule + [SEP_INDEX]
        # add edges
        molecule = molecule + adj_mat_list + [EOS_INDEX]
        # make sure the length of the list is 3 + num_nodes + num_nodes*(num_nodes-1)/2
        molecule_len = 3 + num_nodes + num_nodes*(num_nodes-1)//2
        max_molecule_len = 3 + self.max_nodes + self.max_nodes*(self.max_nodes-1)//2
        assert (len(molecule) == molecule_len)

        # PAD EXTRA AT THE END
        molecule = molecule + [PAD_INDEX] * (max_molecule_len - molecule_len)
        assert (len(molecule) == max_molecule_len)

        # return molecule and correct molecule len (not including padding)
        return [torch.LongTensor(np.array(molecule)), torch.LongTensor(np.array([molecule_len]))]

    def __len__(self):
        return len(self.mol_nodeinds)

class QM9DatasetGraph(TorchDataset):
    def __init__(self, graph_infos, params, train=True, graph_properties=None):
        # set parameters
        self.max_nodes = params.max_nodes
        self.num_node_types = params.num_node_types
        self.num_edge_types = params.num_edge_types
        self.no_edge_present_type = params.no_edge_present_type
        self.do_not_corrupt = params.do_not_corrupt
        self.target_frac_type = params.target_frac_type

        self.node_target_frac = params.node_target_frac
        self.val_node_target_frac = params.val_node_target_frac
        self.node_mask_frac = params.node_mask_frac
        self.node_replace_frac = params.node_replace_frac
        self.node_mask_predict_frac = params.node_mask_predict_frac
        self.node_replace_predict_frac = params.node_replace_predict_frac

        self.edge_target_frac = params.edge_target_frac
        self.val_edge_target_frac = params.val_edge_target_frac
        self.edge_mask_frac = params.edge_mask_frac
        self.edge_replace_frac = params.edge_replace_frac
        self.edge_mask_predict_frac = params.edge_mask_predict_frac
        self.edge_replace_predict_frac = params.edge_replace_predict_frac

        self.mask_round_func = np.ceil if params.force_mask_predict is True else np.round
        self.replace_round_func = np.ceil if params.force_replace_predict is True else np.round

        self.debug_fixed = params.debug_fixed
        self.equalise = params.equalise
        self.seed = params.seed
        self.max_hs = params.max_hs
        self.weighted_loss = params.weighted_loss if hasattr(params, 'weighted_loss') else False

        self.node_properties, self.edge_properties, self.graph_properties = {}, {}, {}

        self.min_charge = params.min_charge; self.max_charge = params.max_charge
        self.mask_all_ring_properties = params.mask_all_ring_properties

        self.node_property_names = ['node_type', 'hydrogens', 'charge', 'is_in_ring', 'is_aromatic', 'chirality']
        self.edge_property_names = ['edge_type']
        self.graph_property_names = ['molwt', 'logp']
        self.normalise_graph_properties = params.normalise_graph_properties
        node_type, edge_type, num_hs, charge, is_in_ring, is_aromatic, chirality = zip(*graph_infos)
        if graph_properties != None:
            self.graph_properties['molwt'], self.graph_properties['logp'] = zip(*graph_properties)
            self.graph_property_stats = {name: {'mean': np.mean(data), 'std': np.std(data)} \
                                         for name, data in self.graph_properties.items()}

        num_h_categories = self.max_hs + 1
        num_charge_categories = abs(self.min_charge) + self.max_charge + 1
        num_is_in_ring_categories = 2
        num_is_aromatic_categories = 2
        num_chirality_categories = 4

        if sparse.issparse(edge_type[0]): edge_type = [et.todense() for et in edge_type]
        # indices -> 0: no hydrogens, max_hs: max_hydrogens, max_hs+1: mask
        h_mask_index = self.max_hs + 1
        h_empty_index = 0
        charge_mask_index = num_charge_categories
        is_in_ring_mask_index = num_is_in_ring_categories
        is_aromatic_mask_index = num_is_aromatic_categories
        chirality_mask_index = num_chirality_categories
        # node arrays are of format [nodes ... , mask]
        # edge arrays are of format [no edge, edge types ... , mask]
        node_mask_index = self.num_node_types
        node_empty_index = self.num_node_types + 1
        edge_mask_index = self.num_edge_types
        if self.no_edge_present_type == 'learned':
            edge_empty_index = self.num_edge_types + 1
        elif self.no_edge_present_type == 'zeros':
            edge_empty_index = 0

        def add_to_dict(dct, property_name, data, num_categories, mask_index, empty_index):
            dct[property_name] = {'data': data, 'num_categories': num_categories, 'mask_index': mask_index,
                                  'empty_index': empty_index}
        add_to_dict(self.node_properties, 'node_type', node_type, self.num_node_types, node_mask_index, node_empty_index)
        add_to_dict(self.node_properties, 'hydrogens', num_hs, num_h_categories, h_mask_index, h_empty_index)
        add_to_dict(self.node_properties, 'charge', charge, num_charge_categories, charge_mask_index, abs(self.min_charge))
        add_to_dict(self.node_properties, 'is_in_ring', is_in_ring, num_is_in_ring_categories, is_in_ring_mask_index, 0)
        add_to_dict(self.node_properties, 'is_aromatic', is_aromatic, num_is_aromatic_categories, is_aromatic_mask_index, 0)
        add_to_dict(self.node_properties, 'chirality', chirality, num_chirality_categories, chirality_mask_index, 0)

        add_to_dict(self.edge_properties, 'edge_type', edge_type, self.num_edge_types, edge_mask_index, edge_empty_index)

        self.node_property_weights, self.edge_property_weights = self.get_loss_weights()

        self.mask_independently = params.mask_independently
        self.target_data_structs = params.target_data_structs
        if self.target_data_structs == 'nodes':
            self.nodes_only, self.edges_only = True, False
        elif self.target_data_structs == 'edges':
            self.nodes_only, self.edges_only = False, True
        elif self.target_data_structs == 'both':
            self.nodes_only, self.edges_only = False, False
        self.prediction_data_structs = params.prediction_data_structs

        self.pad = True if params.edges_per_batch <= 0 else False

        self.train = train

    def get_loss_weights(self):
        # ones instead of zeros to avoid possible divide by zero later on
        node_property_counts, edge_property_counts = {}, {}
        for name, property_info in self.node_properties.items():
            node_property_counts[name] = np.ones(property_info['num_categories'])
            ml = property_info['num_categories']
            for single_datapoint_property in property_info['data']:
                if name == 'charge': single_datapoint_property = single_datapoint_property + abs(self.min_charge)
                node_property_counts[name] += np.bincount(single_datapoint_property, minlength=ml)
        for name, property_info in self.edge_properties.items():
            edge_property_counts[name] = np.ones(property_info['num_categories'])
            ml = property_info['num_categories']
            for single_datapoint_property in property_info['data']:
                flattened_triu_property = np.triu(single_datapoint_property, k=1).flatten().astype(np.int)
                edge_property_counts[name] += np.bincount(flattened_triu_property, minlength=ml)

        node_property_weights = {name: torch.Tensor(counts.max() / counts) \
                                for name, counts in node_property_counts.items()}
        edge_property_weights = {name: torch.Tensor(counts.max() / counts) \
                                for name, counts in edge_property_counts.items()}

        return node_property_weights, edge_property_weights

    def select_nodes_or_edges(self, num_nodes):
        rnd = np.random.rand()
        cutoff = 1 / ((num_nodes + 1) / 2)
        if rnd <= cutoff:
            nodes, edges = True, False
        else:
            nodes, edges = False, True
        return nodes, edges

    def get_orig_and_init_node_property(self, data, max_nodes, num_nodes, empty_index):
        orig = np.ones(max_nodes) * empty_index
        orig[:num_nodes] = data
        init = np.copy(orig)
        return init, orig

    def get_orig_and_init_edge_property(self, data, max_nodes, num_nodes, empty_index):
        orig = np.ones((max_nodes, max_nodes)) * empty_index
        orig[:num_nodes, :num_nodes] = data
        init = np.copy(orig)
        return init, orig

    def get_target_inds(self, unpadded_inds, majority_index, num_components, component_type):
        if self.debug_fixed:
            target_inds = np.array([0])
        elif (component_type == 'node' and self.edges_only) or (component_type == 'edge' and self.nodes_only):
            target_inds = np.array([], dtype=int)
        else:
            if self.train is True:
                # Sample fraction of nodes to use as targets, or use predetermined value
                if self.target_frac_type == 'fixed':
                    target_frac = getattr(self, '{}_target_frac'.format(component_type))
                elif self.target_frac_type == 'random':
                    target_frac = np.random.uniform(0, getattr(self, '{}_target_frac'.format(component_type)))
            else:
                # Use small fixed target fraction for validation
                target_frac = getattr(self, 'val_{}_target_frac'.format(component_type))
            # if we want equal numbers of non-carbon and carbon target nodes,
            # and nodes are not either all carbon or all non-carbon, pick target nodes as follows
            if self.equalise is True and (unpadded_inds != majority_index).sum() != 0 and\
                    (unpadded_inds == majority_index).sum() != 0:
                num_replacements = int(np.ceil(num_components * target_frac))
                num_target_majority_components = max(int(num_replacements / 2), 1)
                majority_component_locs = np.where(unpadded_inds == majority_index)[0]
                num_target_nonmajority_components = max(num_replacements - num_target_majority_components, 1)
                nonmajority_component_locs = np.where(unpadded_inds != majority_index)[0]
                replace = False if num_target_majority_components <= len(majority_component_locs) else True
                target_inds = np.random.choice(majority_component_locs, num_target_majority_components,
                                                    replace=replace)
                replace = False if num_target_nonmajority_components <= len(nonmajority_component_locs) else True
                target_inds = np.concatenate((target_inds, np.random.choice(nonmajority_component_locs,
                                                                                      num_target_nonmajority_components,
                                                                                      replace=replace)
                                                   ))
            else:
                target_inds = np.random.choice(num_components,
                                                    int(np.ceil(num_components * target_frac)), replace=False)
        return target_inds

    def get_target_subtype_inds(self, target_inds, component_type):
        # Pick components to mask, replace and reconstruct based on random sampler output. Bigger probabilities for an action
        # correspond to bigger intervals for that action in the output, so they are more likely to be picked
        mask_frac = getattr(self, '{}_mask_frac'.format(component_type))
        replace_frac = getattr(self, '{}_replace_frac'.format(component_type))
        shuffled_target_inds = np.random.permutation(target_inds)
        num_masked_components = int(self.mask_round_func(mask_frac * len(target_inds)))
        mask_inds = shuffled_target_inds[:num_masked_components]
        num_replaced_components = min(int(self.replace_round_func(replace_frac * len(target_inds))),
                                 len(target_inds) - num_masked_components)
        replace_inds = shuffled_target_inds[num_masked_components:num_masked_components + num_replaced_components]
        recon_inds = shuffled_target_inds[num_masked_components + num_replaced_components:]
        return mask_inds, replace_inds, recon_inds, num_masked_components, num_replaced_components

    def set_edge_values(self, init_edges, inds, all_coords, value_to_set):
        coords = all_coords[inds]
        coords_0, coords_1 = np.array(list(zip(*coords)))
        init_edges[coords_0, coords_1] = value_to_set
        init_edges[coords_1, coords_0] = value_to_set
        return init_edges, coords


    def corrupt_nodes(self, init_nodes, unpadded_node_inds, node_target_types, num_nodes, predict_nodes,
                      mask_index, num_types):
        # node_target_inds: indices corresponding to positions of nodes in node_inds/init_nodes
        # node_mask_inds: indices corresponding to positions of nodes in init_nodes that will be masked
        # node_mask_predict_inds: indices corresponding to positions of nodes in init_nodes that will be masked and
        #                         will be predicted (used for loss computation) during training
        node_target_inds = self.get_target_inds(unpadded_node_inds, 1, num_nodes, 'node')
        node_mask_inds, node_replace_inds, node_recon_inds, num_masked_nodes,\
            num_replaced_nodes = self.get_target_subtype_inds(node_target_inds, 'node')
        if num_masked_nodes > 0:
            # mask nodes
            init_nodes[node_mask_inds] = mask_index
            node_mask_predict_inds = np.random.choice(node_mask_inds, int(self.mask_round_func(
                                                          self.node_mask_predict_frac * num_masked_nodes)),
                                                      replace=False)
            if predict_nodes is True and len(node_mask_predict_inds) > 0:
                node_target_types[node_mask_predict_inds] = 1
        if num_replaced_nodes > 0:
            # replace nodes
            replacement_nodes = np.random.randint(0, num_types, size=num_replaced_nodes)
            node_replace_predict_inds = np.random.choice(num_replaced_nodes, int(self.replace_round_func(
                                                             self.node_replace_predict_frac * num_replaced_nodes)),
                                                         replace=False)
            init_nodes[node_replace_inds] = replacement_nodes
            if predict_nodes is True and len(node_replace_predict_inds) > 0:
                node_target_types[node_replace_predict_inds] = 2
        # Reconstruction nodes don't need to be changed
        if predict_nodes is True and len(node_recon_inds) > 0:
            node_target_types[node_recon_inds] = 3

        return init_nodes, node_target_types, node_mask_inds, node_replace_inds, node_recon_inds

    def corrupt_node_property(self, init, node_mask_inds, node_replace_inds, mask_index, num_categories):
        init[node_mask_inds] = mask_index
        num_replaced_nodes = len(node_replace_inds)
        if num_replaced_nodes > 0:
            # highest possible entry in sample corresponds to all numbers of hydrogens greater than self.max_hs
            replacement_values = np.random.randint(0, num_categories, size=num_replaced_nodes)
            init[node_replace_inds] = replacement_values

    def corrupt_edges(self, init_edges, edge_coords, unpadded_edge_inds, edge_target_types, num_edges, predict_edges):
        edge_target_inds = self.get_target_inds(unpadded_edge_inds, 0, num_edges, 'edge')
        edge_mask_inds, edge_replace_inds, edge_recon_inds, num_masked_edges, \
        num_replaced_edges = self.get_target_subtype_inds(edge_target_inds, 'edge')

        if num_masked_edges > 0:
            # mask edges
            init_edges, edge_mask_coords = self.set_edge_values(init_edges, edge_mask_inds, edge_coords,
                                                                self.edge_properties['edge_type']['mask_index'])
            mask_predict_edges = edge_mask_coords[np.random.choice(len(edge_mask_coords), int(self.mask_round_func(
                                                                    self.edge_mask_predict_frac * num_masked_edges)),
                                                                   replace=False)]
            if predict_edges is True and len(mask_predict_edges) > 0:
                mask_predict_edges_0, mask_predict_edges_1 = np.array(list(zip(*mask_predict_edges)))
                edge_target_types[mask_predict_edges_0, mask_predict_edges_1] = 1

        if num_replaced_edges > 0:
            # replace edges
            replacement_edges = np.random.randint(0, self.edge_properties['edge_type']['num_categories'],
                                                  size=num_replaced_edges)
            init_edges, edge_replace_coords = self.set_edge_values(init_edges, edge_replace_inds, edge_coords,
                                                                   replacement_edges)
            replace_predict_edges = edge_replace_coords[np.random.choice(len(edge_replace_coords),
                                                        int(self.replace_round_func(
                                                        self.edge_replace_predict_frac * num_replaced_edges)),
                                                        replace=False)]
            if predict_edges is True and len(replace_predict_edges) > 0:
                replace_predict_edges_0, replace_predict_edges_1 = np.array(list(zip(*replace_predict_edges)))
                edge_target_types[replace_predict_edges_0, replace_predict_edges_1] = 2

        if predict_edges is True and len(edge_recon_inds) > 0:
            edge_recon_coords = edge_coords[edge_recon_inds]
            edge_recon_coords = np.array(list(zip(*edge_recon_coords)))
            edge_target_types[edge_recon_coords[0], edge_recon_coords[1]] = 3

        return init_edges, edge_target_types

    def corrupt_graph(self, unpadded_node_properties, init_node_properties, node_property_target_types, num_nodes,
                      unpadded_edge_property_inds, init_edge_properties, edge_property_target_types,
                      edge_coords, num_edges):
        if self.target_data_structs == 'random':
            # sample type of graph components (nodes or edges) to be masked
            self.nodes_only, self.edges_only = self.select_nodes_or_edges(num_nodes)

        predict_nodes, predict_edges = True, True
        if self.target_data_structs == 'both':
            if self.prediction_data_structs == 'random':
                predict_nodes, predict_edges = self.select_nodes_or_edges(num_nodes)
            elif self.prediction_data_structs == 'nodes':
                predict_nodes, predict_edges = True, False
            elif self.prediction_data_structs == 'edges':
                predict_nodes, predict_edges = False, True

        for i, property_name in enumerate(unpadded_node_properties.keys()):
            if self.mask_independently is True or i == 0:
                init_node_properties[property_name], node_property_target_types[property_name], node_mask_inds, \
                    node_replace_inds, node_recon_inds = self.corrupt_nodes(
                    init_node_properties[property_name], unpadded_node_properties[property_name],
                    node_property_target_types[property_name], num_nodes, predict_nodes,
                    self.node_properties[property_name]['mask_index'],
                    self.node_properties[property_name]['num_categories']
                    )
                latest_node_property_target_types = node_property_target_types[property_name]
            else:
                self.corrupt_node_property(init_node_properties[property_name], node_mask_inds, node_replace_inds,
                                           self.node_properties[property_name]['mask_index'],
                                           self.node_properties[property_name]['num_categories'])
                node_property_target_types[property_name] = np.copy(latest_node_property_target_types)

        for i, property_name in enumerate(unpadded_edge_property_inds.keys()):
            init_edge_properties[property_name], edge_property_target_types[property_name] = self.corrupt_edges(
                init_edge_properties[property_name], edge_coords, unpadded_edge_property_inds[property_name],
                edge_property_target_types[property_name], num_edges, predict_edges)

        return (init_node_properties, node_property_target_types, init_edge_properties, edge_property_target_types)

    def __getitem__(self, index):
        set_seed_if(self.seed)

        # *** Initialise nodes ***
        unpadded_node_properties, init_node_properties, orig_node_properties, node_property_target_types = {}, {}, {}, {}
        for property_name, property_info in self.node_properties.items():
            unpadded_data = property_info['data'][index]
            if property_name == 'charge': unpadded_data = unpadded_data + abs(self.min_charge)
            unpadded_node_properties[property_name] = unpadded_data
            num_nodes = len(unpadded_data)
            max_nodes = self.max_nodes if self.pad is True else num_nodes
            init_node_properties[property_name], orig_node_properties[property_name] = \
                self.get_orig_and_init_node_property(
                    unpadded_data, max_nodes, len(unpadded_data), property_info['empty_index'])
            node_property_target_types[property_name] = np.zeros(max_nodes)

        # Create masks with 0 where node does not exist or edge would connect non-existent node, 1 everywhere else
        node_mask = torch.zeros(max_nodes)
        node_mask[:num_nodes] = 1
        edge_mask = torch.zeros((max_nodes, max_nodes))
        edge_mask[:num_nodes, :num_nodes] = 1
        edge_mask[np.arange(num_nodes), np.arange(num_nodes)] = 0

        # *** Initialise edges ***
        edge_coords = [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes)]
        num_edges = len(edge_coords)
        unpadded_edge_properties, unpadded_edge_property_inds, init_edge_properties, orig_edge_properties,\
            edge_property_target_types = {}, {}, {}, {}, {}
        for property_name, property_info in self.edge_properties.items():
            unpadded_data = property_info['data'][index]
            unpadded_edge_properties[property_name] = unpadded_data
            assert (check_symmetric(unpadded_data.astype(int)))  # make sure bond matrix is symmetric
            unpadded_edge_property_inds[property_name] = np.array([unpadded_data[i, j] for (i, j) in edge_coords])
            init_edge_properties[property_name], orig_edge_properties[property_name] =\
                self.get_orig_and_init_edge_property(
                unpadded_data, max_nodes, len(unpadded_data), property_info['empty_index'])
            edge_property_target_types[property_name] = np.zeros(edge_mask.shape)
        edge_coords = np.array(edge_coords)

        if self.do_not_corrupt is False:
            init_node_properties, node_property_target_types, init_edge_properties, edge_property_target_types = \
                self.corrupt_graph(unpadded_node_properties, init_node_properties, node_property_target_types,
                                   num_nodes, unpadded_edge_property_inds, init_edge_properties,
                                   edge_property_target_types, edge_coords, num_edges)

        if self.no_edge_present_type == 'zeros':
            edge_mask[np.where(init_edge_properties['edge_type'] == 0)] = 0

        # Cast to suitable type
        """
        init_node_properties = torch.LongTensor(np.stack([init_node_properties[property_name] \
                                                          for property_name in self.node_property_names]))
        orig_node_properties = torch.LongTensor(np.stack([orig_node_properties[property_name] \
                                                          for property_name in self.node_property_names]))
        node_property_target_types = np.stack([node_property_target_types[property_name] \
                                              for property_name in self.node_property_names]).astype(np.int8)

        init_edge_properties = torch.LongTensor(np.stack([init_edge_properties[property_name] \
                                                          for property_name in self.edge_property_names]))
        orig_edge_properties = torch.LongTensor(np.stack([orig_edge_properties[property_name] \
                                                          for property_name in self.edge_property_names]))
        edge_property_target_types = np.stack([edge_property_target_types[property_name] \
                                               for property_name in self.edge_property_names]).astype(np.int8)
        """


        for property_name in init_node_properties.keys():
            init_node_properties[property_name] = torch.LongTensor(init_node_properties[property_name])
            orig_node_properties[property_name] = torch.LongTensor(orig_node_properties[property_name])
            node_property_target_types[property_name] = node_property_target_types[property_name].astype(np.int8)

        for property_name in init_edge_properties.keys():
            init_edge_properties[property_name] = torch.LongTensor(init_edge_properties[property_name])
            orig_edge_properties[property_name] = torch.LongTensor(orig_edge_properties[property_name])
            edge_property_target_types[property_name] = edge_property_target_types[property_name].astype(np.int8)

        graph_properties = {}
        for k, v in self.graph_properties.items():
            if self.normalise_graph_properties is True:
                graph_properties[k] = torch.Tensor([ (v[index] - self.graph_property_stats[k]['mean']) / \
                                                     self.graph_property_stats[k]['std'] ])
            else:
                graph_properties[k] = torch.Tensor([ v[index] ])

        ret_list = [init_node_properties, orig_node_properties, node_property_target_types, node_mask,
                    init_edge_properties, orig_edge_properties, edge_property_target_types, edge_mask,
                    graph_properties]

        return ret_list

    def __len__(self):
        return len(self.node_properties['node_type']['data'])

class SizeSampler(Sampler):
    r"""Samples elements sequentially in order of size.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, edges_per_batch, shuffle=False):
        self.shuffle = shuffle
        self.lengths = np.array([len(nodes)**2 for nodes in data_source.node_properties['node_type']['data']])
        self.indices = np.arange(len(data_source))
        self.indices = self.indices[np.argsort(self.lengths[self.indices], kind='mergesort')]
        batch_ids = []
        current_batch_edges_per_graph, current_batch_id, current_batch_num_edges = 0, 0, 0
        for i, length in enumerate(self.lengths[self.indices]):
            current_batch_num_edges += length
            if current_batch_edges_per_graph < length or current_batch_num_edges > edges_per_batch:
                current_batch_id += 1
                current_batch_edges_per_graph = length
                current_batch_num_edges = 0
            batch_ids.append(current_batch_id)
        _, bounds = np.unique(batch_ids, return_index=True)
        self.batches = [self.indices[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
        if bounds[-1] < len(self.indices):
            self.batches.append(self.indices[bounds[-1]:])

    def __iter__(self):
        if self.shuffle is True:
            np.random.shuffle(self.batches)
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def load_graph_data(params, train_split=0.8):
    val_batch_size = params.batch_size if params.val_batch_size is None else params.val_batch_size
    val_edges_per_batch = params.edges_per_batch if params.val_edges_per_batch is None else params.val_edges_per_batch
    def load_train_val_data(data_path, val_data_path=None):
        with open(data_path, 'rb') as f:
            graph_infos = pickle.load(f)
        split = int(len(graph_infos) * train_split)
        if params.debug_small:
            train_graph_infos = graph_infos[:params.batch_size*params.num_batches]
            if params.validate_on_train is True:
                val_graph_infos = graph_infos[:params.batch_size*params.num_batches]
            else:
                val_graph_infos = graph_infos[split:]
        elif val_data_path is None:
            train_graph_infos = graph_infos[:split]
            val_graph_infos = graph_infos[split:]
        else:
            train_graph_infos = graph_infos
            with open(val_data_path, 'rb') as f:
                val_graph_infos = pickle.load(f)
        return train_graph_infos, val_graph_infos

    if params.graph_type == 'QM9':
        val_data_path, val_graph_properties_path = None, None
    elif params.graph_type == 'ChEMBL':
        val_data_path, val_graph_properties_path = params.val_data_path, params.val_graph_properties_path

    train_graph_infos, val_graph_infos = load_train_val_data(params.data_path, val_data_path)
    train_graph_infos, filtered_train_indices = limited_node_graph_infos(train_graph_infos, params.max_nodes)
    val_graph_infos, filtered_val_indices = limited_node_graph_infos(val_graph_infos, params.max_nodes)
    if len(params.graph_property_names) > 0:
        train_graph_property_infos, val_graph_property_infos = load_train_val_data(
            params.graph_properties_path, val_graph_properties_path)
        train_graph_property_infos = [train_graph_property_infos[idx] for idx in filtered_train_indices]
        val_graph_property_infos = [val_graph_property_infos[idx] for idx in filtered_val_indices]
    else:
        train_graph_property_infos = val_graph_property_infos = None

    if params.val_dataset_size > 0:
        val_graph_infos = val_graph_infos[:params.val_dataset_size]

    train_dataset = QM9DatasetGraph(train_graph_infos, params, train=True, graph_properties=train_graph_property_infos)
    val_dataset = QM9DatasetGraph(val_graph_infos, params, train=False, graph_properties=val_graph_property_infos)
    if hasattr(params, 'val_seed'): val_dataset.seed = params.val_seed
    if params.edges_per_batch > 0:
        train_batch_sampler = SizeSampler(train_dataset, params.edges_per_batch, params.shuffle)
        train_batch_sampler.batches.reverse()
        train_data_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler)
        val_batch_sampler = SizeSampler(val_dataset, val_edges_per_batch)
        val_data_loader = DataLoader(val_dataset, batch_sampler=val_batch_sampler)
    else:
        train_data_loader = DataLoader(train_dataset, params.batch_size, shuffle=params.shuffle)
        val_data_loader = DataLoader(val_dataset, val_batch_size)

    return train_dataset, val_dataset, train_data_loader, val_data_loader

def limited_node_graph_infos(graph_infos, max_nodes):
    restricted_graph_infos, indices = [], []
    for idx, graph_info in enumerate(graph_infos):
        if 1 < len(graph_info[0]) <= max_nodes:
            restricted_graph_infos.append(graph_info)
            indices.append(idx)
    return restricted_graph_infos, indices

def load_smiles_data(params):
    logger.info('============ Smiles data (%s)')

    data = {}
    data_iterator = {}

    for splt in ['train', 'valid', 'test']:

        # no need to load training data for evaluation
        if splt == 'train' and params.eval_only:
            continue

        if params.data_type == 'ChEMBL':
            path = '{}/guacamol_v1_{}.smiles.pth'.format(params.data_path, splt)
        else:
            if splt == 'test':
                continue
            path = '{}/QM9_{}.smiles.pth'.format(params.data_path, splt)
        # load data / update dictionary parameters / update data
        mono_data = load_binarized(path, params)
        set_dico_parameters(params, data, mono_data['dico'])

        # create stream dataset
        data[splt] = Dataset(mono_data['sentences'], mono_data['positions'], params)

        # if there are several processes on the same machine, we can split the dataset
        if splt == 'train' and params.split_data and 1 < params.n_gpu_per_node <= data['mono_stream'][lang][splt].n_batches:
            n_batches = data[splt].n_batches // params.n_gpu_per_node
            a = n_batches * params.local_rank
            b = n_batches * params.local_rank + n_batches
            data[splt].select_data(a, b)

        if splt == 'train':
            data_iterator[splt] = data[splt].get_iterator(shuffle=True, group_by_size=True, n_sentences=-1)
        else:
            data_iterator[splt] = data[splt].get_iterator(shuffle=True)
        logger.info("")

    logger.info("")
    return data, data_iterator