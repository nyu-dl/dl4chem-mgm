import os, pickle, json
from collections import deque

import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F
from guacamol.distribution_matching_generator import DistributionMatchingGenerator
from rdkit import Chem
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data.gen_targets import get_symbol_list
from src.data.loader import SizeSampler
from src.utils import set_seed_if, graph_to_mol, get_index_method, filter_top_k, calculate_graph_properties,\
    dct_to_cuda, dct_to_cpu

if int(tf.__version__.split('.')[0]) <= 1:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)


class MockGenerator(DistributionMatchingGenerator):
    def __init__(self, smiles_list, num_samples_to_generate, train_smiles_list=None, remove_non_novel=False):
        self.smiles_list = smiles_list
        if remove_non_novel is True:
            self.smiles_list = [s for s in self.smiles_list if s not in train_smiles_list]
        self.smiles_list = self.smiles_list[:num_samples_to_generate]

    def generate(self, number_samples):
        smiles_to_return = self.smiles_list[:number_samples]
        self.smiles_list = self.smiles_list[number_samples:] + self.smiles_list[:number_samples]
        return smiles_to_return

class GenDataset(Dataset):
    def __init__(self, dataset, number_samples):
        self.dataset = dataset
        self.number_samples = number_samples

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return self.number_samples

class GraphGenerator(DistributionMatchingGenerator):
    def __init__(self, train_data, model, generation_algorithm, random_init, num_iters, num_sampling_iters, batch_size,
                 edges_per_batch=-1, retrieve_train_graphs=False, local_cpu=False, cp_save_dir=None,
                 set_seed_at_load_iter=False, graph_type='QM9', sample_uniformly=False, mask_comp_to_predict=False,
                 maintain_minority_proportion=False, no_edge_present_type='learned', mask_independently=False,
                 one_property_per_loop=False, checkpointing_period=1, save_period=1, evaluation_period=1,
                 evaluate_finegrained=False, save_finegrained=False, variables_per_gibbs_iteration=1, top_k=-1,
                 save_init=False, cond_property_values={}):
        super().__init__()
        self.model = model
        self.generation_algorithm = generation_algorithm
        self.random_init = random_init
        self.sample_uniformly = sample_uniformly
        self.num_iters = num_iters
        self.num_sampling_iters = num_sampling_iters
        self.num_argmax_iters = self.num_iters - self.num_sampling_iters
        self.train_data = train_data
        self.batch_size = batch_size
        self.edges_per_batch = edges_per_batch
        self.local_cpu = local_cpu
        self.cp_save_dir = cp_save_dir
        self.calculate_length_dist()
        self.get_special_inds()
        self.set_seed_at_load_iter = set_seed_at_load_iter
        self.symbol_list = get_symbol_list(graph_type)[:self.train_data.num_node_types]
        self.retrieve_train_graphs = retrieve_train_graphs
        self.mask_comp_to_predict = mask_comp_to_predict
        self.maintain_minority_proportion = maintain_minority_proportion
        self.no_edge_present_type = no_edge_present_type
        self.mask_independently = mask_independently
        self.one_property_per_loop = one_property_per_loop
        self.index_method = get_index_method()
        self.checkpointing_period = checkpointing_period
        self.save_period = save_period
        self.evaluation_period = evaluation_period
        self.evaluate_finegrained = evaluate_finegrained
        self.save_finegrained = save_finegrained
        self.variables_per_gibbs_iteration = variables_per_gibbs_iteration
        self.top_k = top_k
        self.save_init = save_init
        self.model_forward = self.model_forward_cgvae if self.model.__class__.__name__ == 'CGVAE' \
                                                      else self.model_forward_mgm

        if self.one_property_per_loop is True:
            self.node_property_ints = {'node_type': 1, 'hydrogens': 2, 'charge': 3, 'is_in_ring': 4, 'is_aromatic': 5,
                                       'chirality': 6}
            self.edge_property_ints = {'edge_type': 7}
        else:
            self.node_property_ints = {'node_type': 1, 'hydrogens': 1, 'charge': 1, 'is_in_ring': 1, 'is_aromatic': 1,
                                       'chirality': 1}
            self.edge_property_ints = {'edge_type': 2}

        self.cond_property_values = {k: float(v) for k, v in cond_property_values.items()}

    def generate(self, number_samples):
        load_path, load_iters = get_load_path(self.num_sampling_iters, self.num_argmax_iters, self.cp_save_dir)
        all_init_node_properties, all_init_edge_properties, all_node_masks, all_edge_masks = \
            self.get_all_init_variables(load_path, number_samples)

        if self.set_seed_at_load_iter is True:
            set_seed_if(load_iters)

        retrieve_train_graphs = self.retrieve_train_graphs
        for j in range(load_iters, self.num_iters):
            if j > 0:
                retrieve_train_graphs = False
                if self.generation_algorithm == 'Gibbs':
                    self.train_data.do_not_corrupt = True
            loader = self.get_dataloader(all_init_node_properties, all_node_masks, all_init_edge_properties,
                                         number_samples, retrieve_train_graphs)

            use_argmax = (j >= self.num_sampling_iters)
            all_init_node_properties, all_init_edge_properties, all_node_masks, \
                smiles_list = self.carry_out_iteration(loader, use_argmax)

        return smiles_list

    def generate_with_evaluation(self, num_samples_to_generate, smiles_dataset_path, output_dir,
                                 num_samples_to_evaluate, evaluate_connected_only=False):

        load_path, load_iters = get_load_path(self.num_sampling_iters, self.num_argmax_iters, self.cp_save_dir)
        all_init_node_properties, all_init_edge_properties, all_node_masks, all_edge_masks = \
            self.get_all_init_variables(load_path, num_samples_to_generate)

        if self.save_init is True and self.random_init is True and load_iters == 0:
            # Save smiles representations of initialised molecules
            smiles_list = []
            num_nodes = all_node_masks.sum(-1)
            for i in range(len(all_init_node_properties['node_type'])):
                mol = graph_to_mol({k: v[i][:int(num_nodes[i])].astype(int) \
                                    for k, v in all_init_node_properties.items()},
                                   {k: v[i][:int(num_nodes[i]), :int(num_nodes[i])].astype(int) \
                                    for k, v in all_init_edge_properties.items()},
                                   min_charge=self.train_data.min_charge, symbol_list=self.symbol_list)
                smiles_list.append(Chem.MolToSmiles(mol))
            save_smiles_list(smiles_list, os.path.join(output_dir, 'smiles_0_0.txt'))
            del smiles_list, mol, num_nodes

        if self.set_seed_at_load_iter is True:
            set_seed_if(load_iters)

        retrieve_train_graphs = self.retrieve_train_graphs
        for j in tqdm(range(load_iters, self.num_iters)):
            if j > 0:
                retrieve_train_graphs = False
                if self.generation_algorithm == 'Gibbs':
                    self.train_data.do_not_corrupt = True
            loader = self.get_dataloader(all_init_node_properties, all_node_masks, all_init_edge_properties,
                                         num_samples_to_generate, retrieve_train_graphs)

            use_argmax = (j >= self.num_sampling_iters)
            all_init_node_properties, all_init_edge_properties, all_node_masks,\
                smiles_list = self.carry_out_iteration(loader, use_argmax)

            sampling_iters_completed = min(j + 1, self.num_sampling_iters)
            argmax_iters_completed = max(0, j + 1 - self.num_sampling_iters)
            if (j + 1 - load_iters) % self.checkpointing_period == 0:
                self.save_checkpoints(all_init_node_properties, all_init_edge_properties,
                                  sampling_iters_completed, argmax_iters_completed)

            if (j + 1 - load_iters) % self.save_period == 0 or (self.save_finegrained is True and (j + 1) <= 10):
                smiles_output_path = os.path.join(output_dir, 'smiles_{}_{}.txt'.format(
                    sampling_iters_completed, argmax_iters_completed))
                save_smiles_list(smiles_list, smiles_output_path)

            if (j + 1 - load_iters) % self.evaluation_period == 0 or \
                (self.evaluate_finegrained is True and (j + 1) <= 10):
                json_output_path = os.path.join(output_dir, 'distribution_results_{}_{}.json'.format(
                                                            sampling_iters_completed, argmax_iters_completed))
                evaluate_uncond_generation(MockGenerator(smiles_list, num_samples_to_generate),
                                                    smiles_dataset_path, json_output_path, num_samples_to_evaluate,
                                                    evaluate_connected_only)
                if self.cond_property_values:
                    cond_json_output_path = os.path.join(output_dir, 'cond_results_{}_{}.json'.format(
                                                            sampling_iters_completed, argmax_iters_completed))
                    self.evaluate_cond_generation(smiles_list[:num_samples_to_evaluate], cond_json_output_path)


    def carry_out_iteration(self, loader, use_argmax):
        mols, smiles_list = [], []
        all_final_node_properties = {name: [] for name in self.train_data.node_property_names}
        all_final_edge_properties = {name: [] for name in self.train_data.edge_property_names}
        all_final_node_masks = []
        print('Generator length: {}'.format(len(loader)), flush=True)
        for init_node_properties, orig_node_properties, node_property_target_types, node_mask, \
            init_edge_properties, orig_edge_properties, edge_property_target_types, edge_mask, \
            graph_properties in tqdm(loader):
            if self.local_cpu is False:
                init_node_properties = dct_to_cuda(init_node_properties)
                node_mask = node_mask.cuda()
                init_edge_properties = dct_to_cuda(init_edge_properties)
                edge_mask = edge_mask.cuda()
                graph_properties = dct_to_cuda(graph_properties)

            if self.generation_algorithm == 'gibbs':
                init_node_properties, init_edge_properties = self.carry_out_gibbs_sampling_sweeps(init_node_properties,
                                                                init_edge_properties, node_mask, edge_mask,
                                                                graph_properties, use_argmax)
            elif self.generation_algorithm == 'simultaneous':
                init_node_properties, init_edge_properties = self.sample_simultaneously(init_node_properties,
                                                    init_edge_properties, node_mask, edge_mask,
                                                    node_property_target_types, edge_property_target_types,
                                                    graph_properties, use_argmax)

            init_node_properties = dct_to_cpu(init_node_properties)
            init_edge_properties = dct_to_cpu(init_edge_properties)
            node_mask = node_mask.cpu()
            del edge_mask

            self.append_and_convert_graphs(init_node_properties, init_edge_properties, node_mask,
                                           all_final_node_properties, all_final_edge_properties, all_final_node_masks,
                                           mols, smiles_list)
        
        return all_final_node_properties, all_final_edge_properties, all_final_node_masks, smiles_list

    def get_all_init_variables(self, load_path, number_samples):
        if load_path is not None:
            with open(load_path, 'rb') as f:
                load_info = pickle.load(f)
            all_init_node_properties, all_init_edge_properties = load_info
            all_node_masks = [(node_type != self.train_data.node_properties['node_type']['empty_index']) \
                              for node_type in all_init_node_properties['node_type']]
            all_edge_masks = [(edge_type != self.train_data.edge_properties['edge_type']['empty_index']) \
                              for edge_type in all_init_edge_properties['edge_type']]
        else:
            lengths = self.sample_lengths(number_samples)
            all_init_node_properties, all_init_edge_properties, all_node_masks, all_edge_masks = \
                self.get_masked_variables(lengths, number_samples, self.edges_per_batch <= 0)
        return all_init_node_properties, all_init_edge_properties, all_node_masks, all_edge_masks

    def get_dataloader(self, all_init_node_properties, all_node_masks, all_init_edge_properties, number_samples,
                       retrieve_train_graphs):
        gen_dataset = GenDataset(self.train_data, number_samples)
        if retrieve_train_graphs is False:
            for name, node_property in all_init_node_properties.items():
                data = []
                for i, single_data_property in enumerate(node_property):
                    if name == 'charge': single_data_property -= abs(self.train_data.min_charge)
                    data.append(single_data_property[:int(all_node_masks[i].sum())])
                gen_dataset.dataset.node_properties[name]['data'] = data

            for name, edge_property in all_init_edge_properties.items():
                data = []
                for i, single_data_property in enumerate(edge_property):
                    data.append(single_data_property[:int(all_node_masks[i].sum()), :int(all_node_masks[i].sum())])
                gen_dataset.dataset.edge_properties[name]['data'] = data

        for name, value in self.cond_property_values.items():
            gen_dataset.dataset.graph_properties[name] = np.ones_like(gen_dataset.dataset.graph_properties[name]) \
                                                         * value

        if self.edges_per_batch > 0:
            batch_sampler = SizeSampler(gen_dataset, self.edges_per_batch)
            batch_sampler.batches.reverse()
            loader = DataLoader(gen_dataset, batch_sampler=batch_sampler)
        else:
            loader = DataLoader(gen_dataset, batch_size=self.batch_size)
        return loader

    def carry_out_gibbs_sampling_sweeps(self, init_node_properties, init_edge_properties, node_mask, edge_mask,
                                        graph_properties, use_argmax):
        if self.edges_per_batch > 0:
            max_nodes = len(init_node_properties['node_type'][0])
            max_edges = int(max_nodes * (max_nodes - 1) / 2)
        else:
            max_nodes, max_edges = self.max_nodes, self.max_edges
        num_nodes = node_mask.sum(-1)
        unique_edge_coords, num_unique_edges, generation_arrays, node_property_arrays, edge_property_arrays = \
            self.get_unshuffled_update_order_arrays(len(init_node_properties['node_type']), num_nodes)

        max_num_components = max_nodes + max_edges
        if self.one_property_per_loop is True:
            max_num_components = len(self.node_property_ints.keys()) * max_nodes + \
                                 len(self.edge_property_ints.keys()) * max_edges
        generation_queue = deque(get_shuffled_array(generation_arrays, max_num_components).transpose())
        node_property_queues = {name: self.get_shuffled_queues(arrays, max_nodes) \
                                for name, arrays in node_property_arrays.items()}
        edge_property_queues = {name: self.get_shuffled_queues(arrays, max_edges) \
                                for name, arrays in edge_property_arrays.items()}
        with torch.no_grad():
            while len(generation_queue) > 0:
                next_target_types = [generation_queue.popleft() \
                                     for _ in range(min(len(generation_queue), self.variables_per_gibbs_iteration))]
                next_target_types = np.vstack(next_target_types)

                node_property_update_graphs = {name: np.where(next_target_types == property_int)[1] \
                                               for name, property_int in self.node_property_ints.items()}
                edge_property_update_graphs = {name: np.where(next_target_types == property_int)[1] \
                                               for name, property_int in self.edge_property_ints.items()}

                # replace nodes, node properties and edges
                nodes_to_update = {name: node_property_queues[name][ind].popleft() \
                                        for name, property_update_graphs in node_property_update_graphs.items() \
                                        for ind in property_update_graphs}
                edges_to_update = {name: edge_property_queues[name][ind].popleft() \
                                   for name, property_update_graphs in edge_property_update_graphs.items() \
                                   for ind in property_update_graphs}

                if self.mask_comp_to_predict is True:
                    self.mask_one_entry_per_graph(init_node_properties, init_edge_properties,
                                                  node_property_update_graphs, edge_property_update_graphs,
                                                  nodes_to_update, edges_to_update, node_mask, edge_mask)

                node_property_scores, edge_property_scores, graph_property_scores = self.model_forward(
                    init_node_properties, init_edge_properties, node_mask, edge_mask, graph_properties)

                node_property_preds, edge_property_preds = self.predict_from_scores(node_property_scores,
                                                                                    edge_property_scores, use_argmax)

                init_node_properties, init_edge_properties = self.update_components(
                        nodes_to_update, edges_to_update, init_node_properties, init_edge_properties,
                        node_property_update_graphs, edge_property_update_graphs,
                        node_property_preds, edge_property_preds)

        return init_node_properties, init_edge_properties

    def sample_simultaneously(self, init_node_properties, init_edge_properties, node_mask, edge_mask,
                                node_property_target_types, edge_property_target_types, graph_properties=None,
                                use_argmax=False):
        with torch.no_grad():
            node_property_scores, edge_property_scores, graph_property_scores = self.model_forward(
                init_node_properties, init_edge_properties, node_mask, edge_mask, graph_properties)
        node_property_preds, edge_property_preds = self.predict_from_scores(node_property_scores, edge_property_scores,
                                                                            use_argmax)

        for name, node_property in init_node_properties.items():
            targets = node_property_target_types[name] != 0
            node_property[targets] = node_property_preds[name][targets]

        for name, edge_property in init_edge_properties.items():
            target_coords = np.where(edge_property_target_types[name] != 0)
            edge_property[target_coords] = edge_property_preds[name][target_coords]
            edge_property[target_coords[0], target_coords[2], target_coords[1]] = \
                edge_property_preds[name][target_coords]

        return init_node_properties, init_edge_properties

    def get_unshuffled_update_order_arrays(self, batch_size, num_nodes):
        unique_edge_coords, num_unique_edges, generation_arrays = [], [], []
        node_property_arrays = {name: [] for name in self.node_property_ints.keys()}
        edge_property_arrays = {name: [] for name in self.edge_property_ints.keys()}
        for i in range(batch_size):
            unique_edge_coords.append(list(zip(*np.triu_indices(num_nodes[i], k=1))))
            num_unique_edges.append(len(unique_edge_coords[i]))
            generation_array = []
            for name, npi in self.node_property_ints.items():
                if self.one_property_per_loop is True: # or self.mask_independently is True ?
                    generation_array.extend([npi] * int(num_nodes[i]))
                else:
                    generation_array.extend([self.node_property_ints['node_type']] * int(num_nodes[i]))
                node_property_arrays[name].append(np.arange(int(num_nodes[i])))
            for name, epi in self.edge_property_ints.items():
                generation_array.extend([epi] * int(num_unique_edges[i]))
                edge_property_arrays[name].append(unique_edge_coords[i])
            else:
                generation_array = np.array([self.node_property_ints['node_type']] * int(num_nodes[i]) +
                                            [self.edge_property_ints['edge_type']] * int(num_unique_edges[i]))
            generation_arrays.append(generation_array)
        return unique_edge_coords, num_unique_edges, generation_arrays, node_property_arrays, edge_property_arrays

    def get_shuffled_queues(self, array_to_shuffle, max_num_components):
        shuffled_array = get_shuffled_array(array_to_shuffle, max_num_components)
        queues = [deque(array) for array in shuffled_array]
        return queues

    def mask_one_entry_per_graph(self, init_node_properties, init_edge_properties,
                                 node_property_update_graphs, edge_property_update_graphs,
                                 nodes_to_update, edges_to_update, node_mask, edge_mask):
        for name in nodes_to_update.keys():
            if nodes_to_update[name]:
                init_node_properties[name][node_property_update_graphs[name], nodes_to_update[name]] = \
                    self.train_data.node_properties[name]['mask_index']
                if self.no_edge_present_type == 'zeros':
                    node_mask[node_property_update_graphs[name], nodes_to_update[name]] = 1
        for name in edges_to_update.keys():
            if edges_to_update[name]:
                coords_array = np.array(edges_to_update[name]).transpose()
                init_edge_properties[edge_property_update_graphs[name], coords_array[0], coords_array[1]] = \
                    init_edge_properties[edge_property_update_graphs[name], coords_array[1], coords_array[0]] = \
                    self.train_data.edge_properties['edge_type']['mask_index']
                if self.no_edge_present_type == 'zeros':
                    edge_mask[edge_property_update_graphs[name], coords_array[0], coords_array[1]] = \
                        edge_mask[edge_property_update_graphs[name], coords_array[1], coords_array[0]] = 1

    def model_forward_mgm(self, init_node_properties, init_edge_properties, node_mask, edge_mask,
                          graph_properties=None):
        return self.model(init_node_properties, node_mask, init_edge_properties, edge_mask, graph_properties)

    def model_forward_cgvae(self, init_nodes, init_edges, node_masks, edge_masks, init_hydrogens):
        node_target_inds_vector = getattr(init_nodes == self.node_mask_index, self.index_method)()
        edge_target_coords_matrix = getattr(init_edges == self.edge_mask_index, self.index_method)()
        node_scores, edge_scores, hydrogen_scores, _, _, _, _ = self.model.prior_forward(
            init_nodes, init_edges, node_masks, edge_masks, init_hydrogens,
            node_target_inds_vector, edge_target_coords_matrix
        )
        return node_scores, edge_scores, hydrogen_scores

    def drop_minority_loc_majority_scores(self, init_nodes, node_scores, majority_node_index=1):
        minority_node_locs = np.where(init_nodes != majority_node_index)
        node_scores[list(minority_node_locs) +
                    [np.ones_like(minority_node_locs[0]) * majority_node_index]] = -9999

    def predict_from_scores(self, node_property_scores, edge_property_scores, use_argmax=False):
        if use_argmax is True:
            node_property_preds = {name: torch.argmax(F.softmax(scores, -1), dim=-1) \
                                   for name, scores in node_property_scores.items()}
            edge_property_preds = {name: torch.argmax(F.softmax(scores, -1), dim=-1) \
                                   for name, scores in edge_property_scores.items()}
        else:
            node_property_preds, edge_property_preds = {}, {}
            for name, scores in node_property_scores.items():
                if self.top_k > 0:
                    scores = filter_top_k(scores, self.top_k)
                node_property_preds[name] = torch.distributions.Categorical(F.softmax(scores, -1)).sample()

            for name, scores in edge_property_scores.items():
                if self.top_k > 0:
                    scores = filter_top_k(scores, self.top_k)
                edge_property_preds[name] = torch.distributions.Categorical(F.softmax(scores, -1)).sample()

        return node_property_preds, edge_property_preds

    def update_components(self, nodes_to_update, edges_to_update, init_node_properties, init_edge_properties,
                        node_property_update_graphs, edge_property_update_graphs,
                        node_property_preds, edge_property_preds):
        for name in nodes_to_update.keys():
            if nodes_to_update[name]:
                init_node_properties[name][node_property_update_graphs[name], nodes_to_update[name]] = \
                    node_property_preds[name][node_property_update_graphs[name], nodes_to_update[name]]
        for name in edges_to_update.keys():
            if edges_to_update:
                coords_array = np.array(edges_to_update).transpose()
                init_edge_properties[name][edge_property_update_graphs[name], coords_array[0], coords_array[1]] = \
                    init_edge_properties[edge_property_update_graphs[name], coords_array[1], coords_array[0]] = \
                    edge_property_preds[name][edge_property_update_graphs[name], coords_array[0], coords_array[1]]
        return init_node_properties, init_edge_properties

    def append_and_convert_graphs(self, init_node_properties, init_edge_properties, node_mask,
                                    all_final_node_properties, all_final_edge_properties, all_final_node_masks,
                                    mols, smiles_list):
        num_nodes = node_mask.sum(-1)
        for i in range(len(init_node_properties['node_type'])):
            for name, node_property in init_node_properties.items():
                all_final_node_properties[name].append(node_property[i].numpy())
            for name, edge_property in init_edge_properties.items():
                all_final_edge_properties[name].append(edge_property[i].numpy())
            all_final_node_masks.append(node_mask[i])
            mol = graph_to_mol({k: v[i][:int(num_nodes[i])] for k, v in init_node_properties.items()},
                    {k: v[i][:int(num_nodes[i]), :int(num_nodes[i])] for k, v in init_edge_properties.items()},
                    min_charge=self.train_data.min_charge, symbol_list=self.symbol_list)
            mols.append(mol)
            smiles_list.append(Chem.MolToSmiles(mol))

    def save_checkpoints(self, all_final_node_properties, all_final_edge_properties, num_sampling_iters,
                         num_argmax_iters):
        if self.cp_save_dir is not None:
            save_path = os.path.join(self.cp_save_dir, 'gen_checkpoint_{}_{}.p'.format(
                                     num_sampling_iters, num_argmax_iters))
            with open(save_path, 'wb') as f:
                pickle.dump([all_final_node_properties, all_final_edge_properties], f)

    def calculate_length_dist(self):
        lengths_dict = {}
        for node_type in self.train_data.node_properties['node_type']['data']:
            length = len(node_type)
            if length not in lengths_dict:
                lengths_dict[length] = 1
            else:
                lengths_dict[length] += 1
        # Normalise
        for key in lengths_dict:
            lengths_dict[key] /= len(self.train_data)
        self.length_dist = lengths_dict

    def get_special_inds(self):
        self.max_nodes = self.train_data.max_nodes
        self.max_edges = int(self.max_nodes * (self.max_nodes-1)/2)

    def sample_lengths(self, number_samples=1):
        lengths = np.array(list(self.length_dist.keys()))
        probs = np.array(list(self.length_dist.values()))
        samples = np.random.choice(lengths, number_samples, p=probs)
        return samples

    def get_masked_variables(self, lengths, number_samples, pad=True):
        if pad is True:
            all_init_node_properties, all_init_edge_properties = {}, {}
            for name, property_info in self.train_data.node_properties.items():
                all_init_node_properties[name] = np.ones((number_samples, self.max_nodes)) * \
                                                 property_info['empty_index']
            for name, property_info in self.train_data.edge_properties.items():
                all_init_edge_properties[name] = np.ones((number_samples, self.max_nodes, self.max_nodes)) * \
                                                 property_info['empty_index']
            node_mask = np.zeros((number_samples, self.max_nodes))
            edge_mask = np.zeros((number_samples, self.max_nodes, self.max_nodes))
        else:
            all_init_node_properties = {name: [] for name in self.train_data.node_property_names}
            all_init_edge_properties = {name: [] for name in self.train_data.node_property_names}
            node_mask, edge_mask = [], []
        for sample_num, length in enumerate(lengths):
            if pad is False:
                for name, property_info in self.train_data.node_properties.items():
                    all_init_node_properties[name].append(np.ones(length) * property_info['empty_index'])
                for name, property_info in self.train_data.edge_properties.items():
                    all_init_edge_properties[name].append(np.ones((length, length)) * property_info['empty_index'])
                node_mask.append(np.zeros(length))
                edge_mask.append(np.zeros((length, length)))
            if self.random_init:
                for name, property_info in self.train_data.node_properties.items():
                    if self.sample_uniformly is True:
                        samples = np.random.randint(0, property_info['num_categories'], size=length)
                    else:
                        samples = torch.distributions.Categorical(1/self.train_data.node_property_weights[name]).sample(
                            [length]).numpy()
                    all_init_node_properties[name][sample_num][:length] = samples

                for name, property_info in self.train_data.edge_properties.items():
                    if self.sample_uniformly is True:
                        samples = np.random.randint(0, property_info['num_categories'],
                                                    size=int(length * (length - 1) / 2))
                    else:
                        samples = torch.distributions.Categorical(1/self.train_data.edge_property_weights[name]).sample(
                            [int(length * (length - 1) / 2)]).numpy()
                    rand_edges = deque(samples)
                    for i in range(length):
                        all_init_edge_properties[name][sample_num][i, i] = 0
                        for j in range(i, length):
                            if i != j:
                                all_init_edge_properties[name][sample_num][i, j] = \
                                    all_init_edge_properties[name][sample_num][j, i] = rand_edges.pop()
            else:
                for name, init_node_property in all_init_node_properties.items():
                    init_node_property[sample_num][:length] = self.train_data.node_properties[name]['mask_index']
                for name, init_edge_property in all_init_edge_properties.items():
                    init_edge_property[sample_num][:length, :length] = \
                        self.train_data.edge_properties[name]['mask_index']

            node_mask[sample_num][:length] = 1
            edge_mask[sample_num][:length, :length] = 1
        return all_init_node_properties, all_init_edge_properties, node_mask, edge_mask

    def evaluate_cond_generation(self, smiles_list, json_output_path):
        valid_mols = []
        for s in smiles_list:
            mol = Chem.MolFromSmiles(s)
            if mol is not None: valid_mols.append(mol)

        graph_properties = calculate_graph_properties(valid_mols, self.cond_property_values.keys())
        graph_property_stats = {name: {'mean': np.mean(graph_property), 'median': np.median(graph_property),
                                       'std': np.std(graph_property)}
                                for name, graph_property in graph_properties.items()}
        with open(json_output_path, 'w') as f:
            json.dump(graph_property_stats, f)

def get_load_path(num_sampling_iters, num_argmax_iters, cp_save_dir):
    all_cp_iters = {}
    for fname in os.listdir(cp_save_dir):
        if 'gen_checkpoint' not in fname: continue
        split_fname = os.path.splitext(fname)[0].split('_')
        cp_sampling_iters, cp_argmax_iters = int(split_fname[2]), int(split_fname[3])
        if cp_sampling_iters in all_cp_iters.keys():
            all_cp_iters[cp_sampling_iters].append(cp_argmax_iters)
        else:
            all_cp_iters[cp_sampling_iters] = [cp_argmax_iters]

    if len(all_cp_iters) == 0:
        return None, 0

    cp_max_sampling_iters = max(all_cp_iters.keys())
    sampling_iters_to_load = min(cp_max_sampling_iters, num_sampling_iters)
    if sampling_iters_to_load == num_sampling_iters and sampling_iters_to_load in all_cp_iters.keys():
        argmax_iters_to_load = min(max(all_cp_iters[sampling_iters_to_load]), num_argmax_iters)
    else:
        argmax_iters_to_load = 0
    if sampling_iters_to_load == argmax_iters_to_load == 0:
        return None, 0

    load_path = os.path.join(cp_save_dir,
                             'gen_checkpoint_{}_{}.p'.format(sampling_iters_to_load, argmax_iters_to_load))
    return load_path, sampling_iters_to_load + argmax_iters_to_load


def get_shuffled_array(arrays, length=None):
    """
    :arg
    arrays: list of generation_arrays
    length: length of an output generation array with padding
    :returns
    shuffled_arrays: padded matrix of shape (number of generation arrays, length)
    """
    if type(arrays[0][0]) == tuple:
        shuffled_arrays = np.ones((len(arrays), length), dtype=(int, 2)) * -1
    else:
        shuffled_arrays = np.ones((len(arrays), length)) * -1
    for i, array in enumerate(arrays):
        array = np.random.permutation(array)
        shuffled_arrays[i, :len(array)] = array
    return shuffled_arrays

def save_smiles_list(smiles_list, smiles_output_path):
    with open(smiles_output_path, 'w') as f:
        for smiles in smiles_list:
            f.write(smiles + '\n')

def evaluate_uncond_generation(mock_generator, smiles_dataset_path,
                                json_output_path, num_samples_to_evaluate, evaluate_connected_only=False):
    from guacamol.assess_distribution_learning import _assess_distribution_learning
    if evaluate_connected_only is True:
        mock_generator.smiles_list = [s for s in mock_generator.smiles_list if '.' not in s]
    _assess_distribution_learning(mock_generator, smiles_dataset_path, json_output_file=json_output_path,
                                  benchmark_version='v1', number_samples=num_samples_to_evaluate)