import copy
import os
import pickle
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
from src.utils import set_seed_if, graph_to_mol, get_index_method, filter_top_k

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
        self.mol_nodeinds = self.dataset.mol_nodeinds # for SizeSampler
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
                 save_init=False):
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

        self.node_int = 1
        self.edge_int = 2
        if self.one_property_per_loop is True:
            self.hydrogen_int, self.charge_int, self.is_in_ring_int, self.is_aromatic_int, self.chirality_int = \
                tuple(range(3, 8))
        else:
            self.hydrogen_int = self.charge_int = self.is_in_ring_int = self.is_aromatic_int = self.chirality_int = \
                self.node_int

    def generate(self, number_samples):
        load_path, load_iters = get_load_path(self.num_sampling_iters, self.num_argmax_iters, self.cp_save_dir)
        all_init_nodes, all_init_edges, all_node_masks, all_edge_masks, all_init_hydrogens, all_init_charge, \
        all_init_is_in_ring, all_init_is_aromatic, all_init_chirality = self.get_all_init_variables(load_path,
                                                                                            number_samples)

        if self.set_seed_at_load_iter is True:
            set_seed_if(load_iters)

        retrieve_train_graphs = self.retrieve_train_graphs
        for j in range(load_iters, self.num_iters):
            if j > 0:
                retrieve_train_graphs = False
                if self.generation_algorithm == 'Gibbs':
                    self.train_data.do_not_corrupt = True
            loader = self.get_dataloader(all_init_nodes, all_init_edges, all_node_masks,
                                         all_init_hydrogens, all_init_charge, all_init_is_in_ring, all_init_is_aromatic,
                                         all_init_chirality, number_samples, retrieve_train_graphs)

            use_argmax = (j >= self.num_sampling_iters)
            all_init_nodes, all_init_edges, all_init_hydrogens, all_init_charge, all_init_is_in_ring,\
                all_init_is_aromatic, all_init_chirality, all_node_masks,\
                smiles_list = self.carry_out_iteration(loader, use_argmax)

        return smiles_list

    def generate_with_evaluation(self, num_samples_to_generate, smiles_dataset_path, output_dir,
                                 num_samples_to_evaluate, evaluate_connected_only=False):

        load_path, load_iters = get_load_path(self.num_sampling_iters, self.num_argmax_iters, self.cp_save_dir)
        all_init_nodes, all_init_edges, all_node_masks, all_edge_masks, all_init_hydrogens, all_init_charge, \
        all_init_is_in_ring, all_init_is_aromatic, all_init_chirality = self.get_all_init_variables(load_path,
                                                                                            num_samples_to_generate)

        if self.save_init is True and self.random_init is True and load_iters == 0:
            # Save smiles representations of initialised molecules
            smiles_list = []
            num_nodes = all_node_masks.sum(-1)
            for i in range(len(all_init_nodes)):
                mol = graph_to_mol(all_init_nodes[i, :int(num_nodes[i])].astype(int),
                                   all_init_edges[i, :int(num_nodes[i]), :int(num_nodes[i])].astype(int),
                                   all_init_charge[i, :int(num_nodes[i])].astype(int),
                                   all_init_chirality[i, :int(num_nodes[i])].astype(int),
                                   min_charge=self.min_charge, symbol_list=self.symbol_list)
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
            loader = self.get_dataloader(all_init_nodes, all_init_edges, all_node_masks,
                                         all_init_hydrogens, all_init_charge, all_init_is_in_ring, all_init_is_aromatic,
                                         all_init_chirality, num_samples_to_generate, retrieve_train_graphs)

            use_argmax = (j >= self.num_sampling_iters)
            all_init_nodes, all_init_edges, all_init_hydrogens, all_init_charge, all_init_is_in_ring,\
                all_init_is_aromatic, all_init_chirality, all_node_masks,\
                smiles_list = self.carry_out_iteration(loader, use_argmax)

            sampling_iters_completed = min(j + 1, self.num_sampling_iters)
            argmax_iters_completed = max(0, j + 1 - self.num_sampling_iters)
            if (j + 1 - load_iters) % self.checkpointing_period == 0:
                self.save_checkpoints(all_init_nodes, all_init_edges, all_init_hydrogens, all_init_charge,
                                  all_init_is_in_ring, all_init_is_aromatic, all_init_chirality,
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


    def carry_out_iteration(self, loader, use_argmax):
        mols, smiles_list = [], []
        all_final_nodes, all_final_edges, all_final_hydrogens, all_final_charge, all_final_is_in_ring, \
            all_final_is_aromatic, all_final_chirality, all_final_node_masks = [], [], [], [], [], [], [], []
        print('Generator length: {}'.format(len(loader)), flush=True)
        for init_nodes, init_edges, orig_nodes, orig_edges, node_masks, edge_masks, node_target_types,\
            edge_target_types, init_hydrogens, orig_hydrogens, init_charge, orig_charge, init_is_in_ring,\
            orig_is_in_ring, init_is_aromatic, orig_is_aromatic, init_chirality, orig_chirality, \
            hydrogen_target_types, charge_target_types, is_in_ring_target_types, is_aromatic_target_types, \
            chirality_target_types in tqdm(loader):
            if self.local_cpu is False:
                init_nodes = init_nodes.cuda(); init_edges = init_edges.cuda(); init_hydrogens = init_hydrogens.cuda()
                init_charge = init_charge.cuda(); init_is_in_ring = init_is_in_ring.cuda()
                init_is_aromatic = init_is_aromatic.cuda(); init_chirality = init_chirality.cuda()
                node_masks = node_masks.cuda(); edge_masks = edge_masks.cuda()

            if self.generation_algorithm == 'gibbs':
                init_nodes, init_edges, init_hydrogens, init_charge, init_is_in_ring, init_is_aromatic, init_chirality \
                                                        = self.carry_out_gibbs_sampling_sweeps(init_nodes, init_edges,
                                                        init_hydrogens, init_charge, init_is_in_ring, init_is_aromatic,
                                                        init_chirality, node_masks, edge_masks, use_argmax)
            elif self.generation_algorithm == 'simultaneous':
                init_nodes, init_edges, init_hydrogens, init_charge, init_is_in_ring, init_is_aromatic, init_chirality \
                                                        = self.sample_simultaneously(init_nodes, init_edges,
                                                        init_hydrogens, init_charge, init_is_in_ring, init_is_aromatic,
                                                        init_chirality, node_masks, edge_masks, node_target_types,
                                                        edge_target_types, hydrogen_target_types, charge_target_types,
                                                        is_in_ring_target_types, is_aromatic_target_types,
                                                        chirality_target_types, use_argmax)

            init_nodes = init_nodes.cpu(); init_edges = init_edges.cpu(); init_hydrogens = init_hydrogens.cpu()
            init_charge = init_charge.cpu(); init_is_in_ring = init_is_in_ring.cpu()
            init_is_aromatic = init_is_aromatic.cpu(); init_chirality = init_chirality.cpu()
            node_masks = node_masks.cpu(); edge_masks = edge_masks.cpu()

            self.append_and_convert_graphs(init_nodes, init_edges, init_hydrogens, init_charge, init_is_in_ring,
                        init_is_aromatic, init_chirality, node_masks, all_final_nodes, all_final_edges,
                        all_final_hydrogens, all_final_charge, all_final_is_in_ring, all_final_is_aromatic,
                        all_final_chirality, all_final_node_masks, mols, smiles_list)
        
        return all_final_nodes, all_final_edges, all_final_hydrogens, all_final_charge, all_final_is_in_ring, \
               all_final_is_aromatic, all_final_chirality, all_final_node_masks, smiles_list

    def get_all_init_variables(self, load_path, number_samples):
        if load_path is not None:
            with open(load_path, 'rb') as f:
                load_info = pickle.load(f)
            if self.model.embed_hs is True:
                all_init_nodes, all_init_edges, all_init_hydrogens, all_init_charge, \
                    all_init_is_in_ring, all_init_is_aromatic, all_init_chirality = load_info
            else:
                all_init_nodes, all_init_edges = load_info
            all_node_masks = [(init_nodes != self.node_empty_index) for init_nodes in all_init_nodes]
            all_edge_masks = [(init_edges != self.edge_empty_index) for init_edges in all_init_edges]
        else:
            lengths = self.sample_lengths(number_samples)
            all_init_nodes, all_init_edges, all_node_masks, all_edge_masks, all_init_hydrogens, all_init_charge,\
                all_init_is_in_ring, all_init_is_aromatic, all_init_chirality = self.get_masked_variables(lengths,
                number_samples, self.train_data.num_node_types, self.train_data.num_edge_types, self.edges_per_batch<=0)
        return all_init_nodes, all_init_edges, all_node_masks, all_edge_masks, all_init_hydrogens, all_init_charge,\
                all_init_is_in_ring, all_init_is_aromatic, all_init_chirality

    def get_dataloader(self, all_init_nodes, all_init_edges, all_node_masks, all_init_hydrogens,
                       all_init_charge, all_init_is_in_ring, all_init_is_aromatic, all_init_chirality, number_samples,
                       retrieve_train_graphs):
        gen_dataset = GenDataset(self.train_data, number_samples)
        if retrieve_train_graphs is False:
            gen_dataset.dataset.mol_nodeinds = [init_nodes[:int(all_node_masks[i].sum())]
                                            for i, init_nodes in enumerate(all_init_nodes)]
            gen_dataset.dataset.num_hs = [init_hydrogens[:int(all_node_masks[i].sum())]
                                            for i, init_hydrogens in enumerate(all_init_hydrogens)]
            gen_dataset.dataset.charge = [init_charge[:int(all_node_masks[i].sum())] - abs(self.min_charge)
                                         for i, init_charge in enumerate(all_init_charge)]
            gen_dataset.dataset.is_in_ring = [init_is_in_ring[:int(all_node_masks[i].sum())]
                                         for i, init_is_in_ring in enumerate(all_init_is_in_ring)]
            gen_dataset.dataset.is_aromatic = [init_is_aromatic[:int(all_node_masks[i].sum())]
                                         for i, init_is_aromatic in enumerate(all_init_is_aromatic)]
            gen_dataset.dataset.chirality = [init_chirality[:int(all_node_masks[i].sum())]
                                         for i, init_chirality in enumerate(all_init_chirality)]
            gen_dataset.dataset.adj_mats = [init_edges[:int(all_node_masks[i].sum()), :int(all_node_masks[i].sum())]
                                        for i, init_edges in enumerate(all_init_edges)]
        if self.edges_per_batch > 0:
            batch_sampler = SizeSampler(gen_dataset, self.edges_per_batch)
            batch_sampler.batches.reverse()
            loader = DataLoader(gen_dataset, batch_sampler=batch_sampler)
        else:
            loader = DataLoader(gen_dataset, batch_size=self.batch_size)
        return loader

    def carry_out_gibbs_sampling_sweeps(self, init_nodes, init_edges, init_hydrogens, init_charge, init_is_in_ring,
                                       init_is_aromatic, init_chirality, node_masks, edge_masks, use_argmax):
        init_nodes_copy = copy.deepcopy(init_nodes.cpu())
        if self.edges_per_batch > 0:
            max_nodes = len(init_nodes[0])
            max_edges = int(max_nodes * (max_nodes - 1) / 2)
        else:
            max_nodes, max_edges = self.max_nodes, self.max_edges
        num_nodes = node_masks.sum(-1)
        unique_edge_coords, num_unique_edges, generation_arrays, nodes_arrays, edges_arrays = \
            self.get_unshuffled_update_order_arrays(len(init_nodes), num_nodes)

        max_num_components = max_nodes + max_edges
        if self.one_property_per_loop is True:
            num_properties = 5
            max_num_components += num_properties * max_nodes
        generation_queue = deque(get_shuffled_array(generation_arrays, max_num_components).transpose())
        nodes_queues = self.get_shuffled_queues(nodes_arrays, max_nodes)
        edges_queues = self.get_shuffled_queues(edges_arrays, max_edges)
        if self.mask_independently is True:
            hydrogen_queues = self.get_shuffled_queues(nodes_arrays, max_nodes)
            charge_queues = self.get_shuffled_queues(nodes_arrays, max_nodes)
            is_in_ring_queues = self.get_shuffled_queues(nodes_arrays, max_nodes)
            is_aromatic_queues = self.get_shuffled_queues(nodes_arrays, max_nodes)
            chirality_queues = self.get_shuffled_queues(nodes_arrays, max_nodes)
        with torch.no_grad():
            while len(generation_queue) > 0:
                next_target_types = [generation_queue.popleft() \
                                     for _ in range(min(len(generation_queue), self.variables_per_gibbs_iteration))]
                next_target_types = np.vstack(next_target_types)

                node_update_graphs = np.where(next_target_types == self.node_int)[1]
                edge_update_graphs = np.where(next_target_types == self.edge_int)[1]

                # replace nodes, node properties and edges
                nodes_to_update = [nodes_queues[ind].popleft() for ind in node_update_graphs]
                edges_to_update = [edges_queues[ind].popleft() for ind in edge_update_graphs]
                if self.mask_independently is True:
                    hydrogen_update_graphs = np.where(next_target_types == self.hydrogen_int)[1]
                    charge_update_graphs = np.where(next_target_types == self.charge_int)[1]
                    is_in_ring_update_graphs = np.where(next_target_types == self.is_in_ring_int)[1]
                    is_aromatic_update_graphs = np.where(next_target_types == self.is_aromatic_int)[1]
                    chirality_update_graphs = np.where(next_target_types == self.chirality_int)[1]
                    hydrogens_to_update = [hydrogen_queues[ind].popleft() for ind in hydrogen_update_graphs]
                    charge_to_update = [charge_queues[ind].popleft() for ind in charge_update_graphs]
                    is_in_ring_to_update = [is_in_ring_queues[ind].popleft() for ind in is_in_ring_update_graphs]
                    is_aromatic_to_update = [is_aromatic_queues[ind].popleft() for ind in is_aromatic_update_graphs]
                    chirality_to_update = [chirality_queues[ind].popleft() for ind in chirality_update_graphs]
                else:
                    hydrogens_to_update = charge_to_update = is_in_ring_to_update = is_aromatic_to_update = \
                        chirality_to_update = nodes_to_update

                if self.mask_comp_to_predict is True:
                    self.mask_one_entry_per_graph(init_nodes, init_edges, init_hydrogens,
                                                  init_charge, init_is_in_ring, init_is_aromatic, init_chirality,
                                                  node_update_graphs, edge_update_graphs,
                                                  hydrogen_update_graphs, charge_update_graphs,
                                                  is_in_ring_update_graphs, is_aromatic_update_graphs,
                                                  chirality_update_graphs,
                                                  nodes_to_update, edges_to_update,
                                                  hydrogens_to_update, charge_to_update, is_in_ring_to_update,
                                                  is_aromatic_to_update, chirality_to_update,
                                                  node_masks, edge_masks)

                node_scores, edge_scores, hydrogen_scores, charge_scores, is_in_ring_scores, is_aromatic_scores, \
                                        chirality_scores = self.model_forward(init_nodes, init_edges,
                                                            node_masks, edge_masks, init_hydrogens, init_charge,
                                                            init_is_in_ring, init_is_aromatic, init_chirality)

                if self.maintain_minority_proportion is True:
                    self.drop_minority_loc_majority_scores(init_nodes_copy, node_scores)

                node_preds, edge_preds, hydrogen_preds, charge_preds, is_in_ring_preds, is_aromatic_preds,\
                    chirality_preds = self.predict_from_scores(node_scores, edge_scores, hydrogen_scores,
                                        charge_scores, is_in_ring_scores, is_aromatic_scores, chirality_scores,
                                        use_argmax)

                init_nodes, init_edges, init_hydrogens, \
                    init_charge, init_is_in_ring, init_is_aromatic, init_chirality, = self.update_components(
                        nodes_to_update, edges_to_update, hydrogens_to_update, charge_to_update,
                        is_in_ring_to_update, is_aromatic_to_update, chirality_to_update,
                        init_nodes, init_edges, init_hydrogens,
                        init_charge, init_is_in_ring, init_is_aromatic, init_chirality,
                        node_update_graphs, edge_update_graphs, hydrogen_update_graphs, charge_update_graphs,
                        is_in_ring_update_graphs, is_aromatic_update_graphs, chirality_update_graphs,
                        node_preds, edge_preds, hydrogen_preds,
                        charge_preds, is_in_ring_preds, is_aromatic_preds, chirality_preds)

        return init_nodes, init_edges, init_hydrogens, init_charge, init_is_in_ring, init_is_aromatic, init_chirality

    def sample_simultaneously(self, init_nodes, init_edges, init_hydrogens, init_charge, init_is_in_ring,
                              init_is_aromatic, init_chirality, node_masks, edge_masks, node_target_types,
                              edge_target_types, hydrogen_target_types, charge_target_types, is_in_ring_target_types,
                              is_aromatic_target_types, chirality_target_types, use_argmax):
        with torch.no_grad():
            node_scores, edge_scores, hydrogen_scores, charge_scores, is_in_ring_scores, is_aromatic_scores,\
                chirality_scores, = self.model_forward(init_nodes, init_edges, node_masks,
                                    edge_masks, init_hydrogens, init_charge, init_is_in_ring, init_is_aromatic,
                                    init_chirality)
        node_preds, edge_preds, hydrogen_preds, charge_preds, is_in_ring_preds, is_aromatic_preds,\
                    chirality_preds = self.predict_from_scores(node_scores, edge_scores, hydrogen_scores,
                                    charge_scores, is_in_ring_scores, is_aromatic_scores, chirality_scores, use_argmax)

        init_nodes[node_target_types != 0] = node_preds[node_target_types != 0]
        edge_target_coords = np.where(edge_target_types != 0)
        init_edges[edge_target_coords] = edge_preds[edge_target_coords]
        init_edges[edge_target_coords[0], edge_target_coords[2], edge_target_coords[1]] = edge_preds[edge_target_coords]
        init_hydrogens[hydrogen_target_types != 0] = hydrogen_preds[hydrogen_target_types != 0]
        init_charge[charge_target_types != 0] = charge_preds[charge_target_types != 0]
        init_is_in_ring[is_in_ring_target_types != 0] = is_in_ring_preds[is_in_ring_target_types != 0]
        init_is_aromatic[is_aromatic_target_types != 0] = is_aromatic_preds[is_aromatic_target_types != 0]
        init_chirality[chirality_target_types != 0] = chirality_preds[chirality_target_types != 0]

        return init_nodes, init_edges, init_hydrogens, init_charge, init_is_in_ring, init_is_aromatic, init_chirality

    def get_unshuffled_update_order_arrays(self, batch_size, num_nodes):
        unique_edge_coords, num_unique_edges, generation_arrays, nodes_arrays, edges_arrays = [], [], [], [], []
        for i in range(batch_size):
            unique_edge_coords.append(list(zip(*np.triu_indices(num_nodes[i], k=1))))
            num_unique_edges.append(len(unique_edge_coords[i]))
            if self.one_property_per_loop is True:
                generation_array = np.array([self.node_int] * int(num_nodes[i]) +
                                [self.edge_int] * int(num_unique_edges[i]) +
                                [self.hydrogen_int] * int(num_nodes[i]) + [self.charge_int] * int(num_nodes[i]) +
                                [self.is_in_ring_int] * int(num_nodes[i]) + [self.is_aromatic_int] * int(num_nodes[i]) +
                                [self.chirality_int] * int(num_nodes[i]))
            else:
                generation_array = np.array([self.node_int] * int(num_nodes[i]) +
                                            [self.edge_int] * int(num_unique_edges[i]))
            generation_arrays.append(generation_array)
            nodes_arrays.append(np.arange(int(num_nodes[i])))
            edges_arrays.append(unique_edge_coords[i])
        return unique_edge_coords, num_unique_edges, generation_arrays, nodes_arrays, edges_arrays

    def get_shuffled_queues(self, array_to_shuffle, max_num_components):
        shuffled_array = get_shuffled_array(array_to_shuffle, max_num_components)
        queues = [deque(array) for array in shuffled_array]
        return queues

    def mask_one_entry_per_graph(self, init_nodes, init_edges, init_hydrogens,
                                 init_charge, init_is_in_ring, init_is_aromatic, init_chirality,
                                 node_update_graphs, edge_update_graphs, hydrogen_update_graphs, charge_update_graphs,
                                 is_in_ring_update_graphs, is_aromatic_update_graphs, chirality_update_graphs,
                                 nodes_to_update, edges_to_update, hydrogens_to_update, charge_to_update,
                                 is_in_ring_to_update, is_aromatic_to_update, chirality_to_update,
                                 node_masks, edge_masks):
        if nodes_to_update:
            init_nodes[node_update_graphs, nodes_to_update] = self.node_mask_index
            if self.no_edge_present_type == 'zeros':
                node_masks[node_update_graphs, nodes_to_update] = 1
        if hydrogens_to_update:
            init_hydrogens[hydrogen_update_graphs, hydrogens_to_update] = self.h_mask_index
        if charge_to_update:
            init_charge[charge_update_graphs, charge_to_update] = self.charge_mask_index
        if is_in_ring_to_update:
            init_is_in_ring[is_in_ring_update_graphs, is_in_ring_to_update] = self.is_in_ring_mask_index
        if is_aromatic_to_update:
            init_is_aromatic[is_aromatic_update_graphs, is_aromatic_to_update] = self.is_aromatic_mask_index
        if chirality_to_update:
            init_chirality[chirality_update_graphs, chirality_to_update] = self.chirality_mask_index
        if edges_to_update:
            coords_array = np.array(edges_to_update).transpose()
            init_edges[edge_update_graphs, coords_array[0], coords_array[1]] = init_edges[edge_update_graphs,
                                                                                        coords_array[1], coords_array[
                                                                                        0]] = self.edge_mask_index
            if self.no_edge_present_type == 'zeros':
                edge_masks[edge_update_graphs, coords_array[0], coords_array[1]] = edge_masks[edge_update_graphs,
                                                                coords_array[0], coords_array[1]] = 1

    def model_forward_mgm(self, init_nodes, init_edges, node_masks, edge_masks, init_hydrogens, init_charge,
                                init_is_in_ring, init_is_aromatic, init_chirality):
        node_scores, edge_scores, hydrogen_scores, charge_scores, is_in_ring_scores, is_aromatic_scores, \
                chirality_scores = self.model(init_nodes, init_edges, node_masks, edge_masks, init_hydrogens,
                                              init_charge, init_is_in_ring, init_is_aromatic, init_chirality)
        return node_scores, edge_scores, hydrogen_scores, charge_scores, is_in_ring_scores, is_aromatic_scores, \
                chirality_scores

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

    def predict_from_scores(self, node_scores, edge_scores, hydrogen_scores, charge_scores, is_in_ring_scores,
                            is_aromatic_scores, chirality_scores,use_argmax=False):
        if use_argmax is True:
            node_preds = torch.argmax(F.softmax(node_scores, -1), dim=-1)
            edge_preds = torch.argmax(F.softmax(edge_scores, -1), dim=-1)
            hydrogen_preds = torch.argmax(F.softmax(hydrogen_scores, -1), dim=-1)
            charge_preds = torch.argmax(F.softmax(charge_scores, -1), dim=-1)
            is_in_ring_preds = torch.argmax(F.softmax(is_in_ring_scores, -1), dim=-1)
            is_aromatic_preds = torch.argmax(F.softmax(is_aromatic_scores, -1), dim=-1)
            chirality_preds = torch.argmax(F.softmax(chirality_scores, -1), dim=-1)
        else:
            if self.top_k > 0:
                node_scores = filter_top_k(node_scores, self.top_k)
                edge_scores = filter_top_k(edge_scores, self.top_k)
                hydrogen_scores = filter_top_k(hydrogen_scores, self.top_k)
                charge_scores = filter_top_k(charge_scores, self.top_k)
                is_in_ring_scores = filter_top_k(is_in_ring_scores, self.top_k)
                is_aromatic_scores = filter_top_k(is_aromatic_scores, self.top_k)
                chirality_scores = filter_top_k(chirality_scores, self.top_k)

            node_preds = torch.distributions.Categorical(F.softmax(node_scores, -1)).sample()
            edge_preds = torch.distributions.Categorical(F.softmax(edge_scores, -1)).sample()
            hydrogen_preds = torch.distributions.Categorical(F.softmax(hydrogen_scores, -1)).sample()
            charge_preds = torch.distributions.Categorical(F.softmax(charge_scores, -1)).sample()
            is_in_ring_preds = torch.distributions.Categorical(F.softmax(is_in_ring_scores, -1)).sample()
            is_aromatic_preds = torch.distributions.Categorical(F.softmax(is_aromatic_scores, -1)).sample()
            chirality_preds = torch.distributions.Categorical(F.softmax(chirality_scores, -1)).sample()
        return node_preds, edge_preds, hydrogen_preds, charge_preds, is_in_ring_preds, is_aromatic_preds, \
               chirality_preds

    def update_components(self, nodes_to_update, edges_to_update, hydrogens_to_update,
                          charge_to_update, is_in_ring_to_update,  is_aromatic_to_update, chirality_to_update,
                          init_nodes, init_edges, init_hydrogens,
                          init_charge, init_is_in_ring, init_is_aromatic, init_chirality,
                          node_update_graphs, edge_update_graphs, hydrogen_update_graphs, charge_update_graphs,
                          is_in_ring_update_graphs, is_aromatic_update_graphs, chirality_update_graphs,
                          node_preds, edge_preds, hydrogen_preds,
                          charge_preds, is_in_ring_preds, is_aromatic_preds, chirality_preds):
        if nodes_to_update:
            init_nodes[node_update_graphs, nodes_to_update] = node_preds[node_update_graphs, nodes_to_update]
        if hydrogens_to_update:
            init_hydrogens[hydrogen_update_graphs, hydrogens_to_update] = \
                hydrogen_preds[hydrogen_update_graphs, hydrogens_to_update]
        if charge_to_update:
            init_charge[charge_update_graphs, charge_to_update] = charge_preds[charge_update_graphs, charge_to_update]
        if is_in_ring_to_update:
            init_is_in_ring[is_in_ring_update_graphs, is_in_ring_to_update] = is_in_ring_preds[
                is_in_ring_update_graphs, is_in_ring_to_update]
        if is_aromatic_to_update:
            init_is_aromatic[is_aromatic_update_graphs, is_aromatic_to_update] = is_aromatic_preds[
                is_aromatic_update_graphs, is_aromatic_to_update]
        if chirality_to_update:
            init_chirality[chirality_update_graphs, chirality_to_update] = chirality_preds[
                chirality_update_graphs, chirality_to_update]
        if edges_to_update:
            coords_array = np.array(edges_to_update).transpose()
            init_edges[edge_update_graphs, coords_array[0], coords_array[1]] = init_edges[
                edge_update_graphs,
                coords_array[1], coords_array[0]] = edge_preds[
                edge_update_graphs, coords_array[0], coords_array[1]]
        return init_nodes, init_edges, init_hydrogens, init_charge, init_is_in_ring, init_is_aromatic, init_chirality

    def append_and_convert_graphs(self, init_nodes, init_edges, init_hydrogens, init_charge, init_is_in_ring,
                                  init_is_aromatic, init_chirality, node_masks, all_final_nodes, all_final_edges,
                                  all_final_hydrogens, all_final_charge, all_final_is_in_ring, all_final_is_aromatic,
                                  all_final_chirality, all_final_node_masks, mols, smiles_list):
        init_nodes, init_edges, init_hydrogens = init_nodes.numpy(), init_edges.numpy(), init_hydrogens.numpy()
        num_nodes = node_masks.sum(-1)
        for i in range(len(init_nodes)):
            all_final_nodes.append(init_nodes[i])
            all_final_edges.append(init_edges[i])
            all_final_node_masks.append(node_masks[i])
            if self.model.embed_hs is True:
                all_final_hydrogens.append(init_hydrogens[i])
                all_final_charge.append(init_charge[i])
                all_final_is_in_ring.append(init_is_in_ring[i])
                all_final_is_aromatic.append(init_is_aromatic[i])
                all_final_chirality.append(init_chirality[i])
            mol = graph_to_mol(init_nodes[i, :int(num_nodes[i])], init_edges[i, :int(num_nodes[i]), :int(num_nodes[i])],
                               init_charge[i, :int(num_nodes[i])], init_chirality[i, :int(num_nodes[i])],
                               min_charge=self.min_charge, symbol_list=self.symbol_list)
            mols.append(mol)
            smiles_list.append(Chem.MolToSmiles(mol))

    def save_checkpoints(self, all_final_nodes, all_final_edges, all_final_hydrogens, all_final_charge,
                         all_final_is_in_ring, all_final_is_aromatic, all_final_chirality, num_sampling_iters,
                         num_argmax_iters):
        if self.cp_save_dir is not None:
            to_save = [all_final_nodes, all_final_edges]
            if self.model.embed_hs is True:
                to_save.extend([all_final_hydrogens, all_final_charge, all_final_is_in_ring, all_final_is_aromatic,
                               all_final_chirality])
            save_path = os.path.join(self.cp_save_dir, 'gen_checkpoint_{}_{}.p'.format(
                                     num_sampling_iters, num_argmax_iters))
            with open(save_path, 'wb') as f:
                pickle.dump(to_save, f)

    def calculate_length_dist(self):
        lengths_dict = {}
        for mol_nodeind in self.train_data.mol_nodeinds:
            length = len(mol_nodeind)
            if length not in lengths_dict:
                lengths_dict[length] = 1
            else:
                lengths_dict[length] += 1
        # Normalise
        for key in lengths_dict:
            lengths_dict[key] /= len(self.train_data)
        self.length_dist = lengths_dict

    def get_special_inds(self):
        self.node_empty_index = self.train_data.node_empty_index
        self.edge_empty_index = self.train_data.edge_empty_index
        self.node_mask_index = self.train_data.node_mask_index
        self.edge_mask_index = self.train_data.edge_mask_index
        self.max_nodes = self.train_data.max_nodes
        self.max_edges = int(self.max_nodes * (self.max_nodes-1)/2)
        if self.train_data.num_hs is not None:
            self.h_mask_index = self.train_data.h_mask_index
            self.h_empty_index = self.train_data.h_empty_index
            self.charge_mask_index = self.train_data.charge_mask_index
            self.is_in_ring_mask_index = self.train_data.is_in_ring_mask_index
            self.is_aromatic_mask_index = self.train_data.is_aromatic_mask_index
            self.chirality_mask_index = self.train_data.chirality_mask_index
            self.min_charge = self.train_data.min_charge
        else:
            self.h_mask_index = self.h_empty_index = 0

    def sample_lengths(self, number_samples=1):
        lengths = np.array(list(self.length_dist.keys()))
        probs = np.array(list(self.length_dist.values()))
        samples = np.random.choice(lengths, number_samples, p=probs)
        return samples

    def get_masked_variables(self, lengths, number_samples, num_node_types, num_edge_types, pad=True):
        if pad is True:
            init_nodes = np.ones((number_samples, self.max_nodes)) * self.node_empty_index
            node_mask = np.zeros((number_samples, self.max_nodes))
            init_edges = np.ones((number_samples, self.max_nodes, self.max_nodes)) * self.edge_empty_index
            edge_mask = np.zeros((number_samples, self.max_nodes, self.max_nodes))
            hydrogens = np.ones((number_samples, self.max_nodes)) * self.h_empty_index
            init_charge = np.ones((number_samples, self.max_nodes)) * abs(self.min_charge)
            init_is_in_ring = np.zeros((number_samples, self.max_nodes))
            init_is_aromatic = np.zeros((number_samples, self.max_nodes))
            init_chirality = np.zeros((number_samples, self.max_nodes))
        else:
            init_nodes, node_mask, init_edges, edge_mask, hydrogens, init_charge, init_is_in_ring, init_is_aromatic, \
                        init_chirality = [], [], [], [], [], [], [], [], []
        for sample_num, length in enumerate(lengths):
            if pad is False:
                init_nodes.append(np.ones(length) * self.node_empty_index)
                node_mask.append(np.zeros(length))
                init_edges.append(np.ones((length, length)) * self.edge_empty_index)
                edge_mask.append(np.zeros((length, length)))
                hydrogens.append(np.ones(length) * self.h_empty_index)
                init_charge.append(np.ones(length) * abs(self.min_charge))
                init_is_in_ring.append(np.zeros(length))
                init_is_aromatic.append(np.zeros(length))
                init_chirality.append(np.zeros(length))
            if self.random_init:
                if self.sample_uniformly is True:
                    node_samples = np.random.randint(0, num_node_types, size=length)
                    edge_samples = np.random.randint(0, num_edge_types, size=int(length * (length - 1) / 2))
                    hydrogen_samples = np.random.randint(0, self.h_mask_index, size=length)
                    charge_samples = np.random.randint(0, self.charge_mask_index, size=length)
                    is_in_ring_samples = np.random.randint(0, self.is_in_ring_mask_index, size=length)
                    is_aromatic_samples = np.random.randint(0, self.is_aromatic_mask_index, size=length)
                    chirality_samples = np.random.randint(0, self.chirality_mask_index, size=length)
                else:
                    node_samples = torch.distributions.Categorical(1/self.train_data.node_weights).sample(
                                                                                                    [length]).numpy()
                    edge_samples = torch.distributions.Categorical(1/self.train_data.edge_weights).sample(
                                                                            [int(length * (length - 1) / 2)]).numpy()
                    hydrogen_samples = torch.distributions.Categorical(1/self.train_data.h_weights).sample(
                                                                                                    [length]).numpy()
                    charge_samples = torch.distributions.Categorical(1/self.train_data.charge_weights).sample(
                                                                                                    [length]).numpy()
                    is_in_ring_samples = torch.distributions.Categorical(1/self.train_data.is_in_ring_weights).sample(
                                                                                                    [length]).numpy()
                    is_aromatic_samples = torch.distributions.Categorical(1/self.train_data.is_aromatic_weights).sample(
                                                                                                    [length]).numpy()
                    chirality_samples = torch.distributions.Categorical(1/self.train_data.chirality_weights).sample(
                                                                                                    [length]).numpy()
                init_nodes[sample_num][:length] = node_samples
                rand_edges = deque(edge_samples)
                for i in range(length):
                    init_edges[sample_num][i, i] = 0
                    for j in range(i, length):
                        if i != j:
                            init_edges[sample_num][i, j] = init_edges[sample_num][j, i] = rand_edges.pop()
                hydrogens[sample_num][:length] = hydrogen_samples
                init_charge[sample_num][:length] = charge_samples
                init_is_in_ring[sample_num][:length] = is_in_ring_samples
                init_is_aromatic[sample_num][:length] = is_aromatic_samples
                init_chirality[sample_num][:length] = chirality_samples
            else:
                init_nodes[sample_num][:length] = self.node_mask_index
                init_edges[sample_num][:length, :length] = self.edge_mask_index
                hydrogens[sample_num][:length] = self.h_mask_index
                init_charge[sample_num][:length] = self.charge_mask_index
                init_is_in_ring[sample_num][:length] = self.is_in_ring_mask_index
                init_is_aromatic[sample_num][:length] = self.is_aromatic_mask_index
                init_chirality[sample_num][:length] = self.chirality_mask_index
            node_mask[sample_num][:length] = 1
            edge_mask[sample_num][:length, :length] = 1
        return init_nodes, init_edges, node_mask, edge_mask, hydrogens, init_charge, init_is_in_ring,\
               init_is_aromatic, init_chirality

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