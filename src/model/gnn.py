import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from src.model.mpnns import MPNN_DICT
from src.utils import IUPAC_VOCAB, DEEPFRI_VOCAB_KEYS, copy_graph_remove_data
from collections import OrderedDict


class GraphNNParent(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.max_nodes = params.max_nodes
        self.dim_h = params.dim_h
        self.dim_k = params.dim_k

        self.no_edge_present_type = params.no_edge_present_type
        self.mpnn_name = params.mpnn_name

        self.node_property_classes_dct = OrderedDict([
                                    ('node_type', {'num_input_classes': params.num_node_types + 2,
                                                    'padding_idx': 0,
                                                    'num_output_classes': params.num_node_types})])
        if params.graph_type in ['QM9', 'ChEMBL']:
            self.node_property_classes_dct.update([
                                    ('hydrogens', {'num_input_classes': params.max_hs + 2,
                                                    'padding_idx': 0,
                                                    'num_output_classes': params.max_hs + 1}),
                                    ('charge', {'num_input_classes': abs(params.min_charge) + params.max_charge + 2,
                                                 'padding_idx': abs(params.min_charge),
                                                 'num_output_classes': abs(params.min_charge) + params.max_charge + 1}),
                                    ('is_in_ring', {'num_input_classes': 3, 'padding_idx': 0,
                                                    'num_output_classes': 2}),
                                    ('is_aromatic', {'num_input_classes': 3, 'padding_idx': 0,
                                                     'num_output_classes': 2}),
                                    ('chirality', {'num_input_classes': 5, 'padding_idx': 0,
                                                    'num_output_classes': 4})
                                    ])

        edge_type_padding_idx = 0 if params.no_edge_present_type == 'zeros' else params.num_edge_types + 1
        self.edge_property_classes_dct = OrderedDict([
            ('edge_type', {'num_input_classes': params.num_edge_types + 2,
                           'padding_idx': edge_type_padding_idx,
                           'num_output_classes': params.num_edge_types}),
            ])

        self.node_embedding_layers = nn.ModuleDict({property_name:
            nn.Embedding(classes_info['num_input_classes'], self.dim_h, padding_idx=classes_info['padding_idx'])
            for property_name, classes_info in self.node_property_classes_dct.items()
            })

        self.edge_embedding_layers = nn.ModuleDict({property_name:
            nn.Embedding(classes_info['num_input_classes'], self.dim_h * self.dim_k,
                         padding_idx=classes_info['padding_idx'])
            for property_name, classes_info in self.edge_property_classes_dct.items()
            })

        self.res_conn = params.res_conn
        self.mpnn_steps = params.mpnn_steps
        self.use_layer_norm = params.layer_norm
        self.update_edges_at_end_only = params.update_edges_at_end_only
        self.use_newest_edges = True if params.use_newest_edges is True else False
        self.node_mpnn_name = params.node_mpnn_name
        self.fully_connected = params.fully_connected
        self.spatial_msg_res_conn = params.spatial_msg_res_conn
        self.spatial_postgru_res_conn = params.spatial_postgru_res_conn
        self.global_connection = params.global_connection
        self.bound_edges = params.bound_edges

        self.nph_dim = int(self.dim_h/2)
        self.node_property_classifiers = nn.ModuleDict({property_name: \
            nn.Sequential(nn.Linear(self.dim_h, self.nph_dim),
                          nn.ReLU(),
                          nn.Linear(self.nph_dim, classes_info['num_output_classes'])) \
            for property_name, classes_info in self.node_property_classes_dct.items()
        })

        self.edge_property_classifiers = nn.ModuleDict({property_name: \
            nn.Sequential(nn.Linear(self.dim_h * self.dim_k, int(self.dim_h * self.dim_k/2)),
                          nn.ReLU(),
                          nn.Linear(int(self.dim_h * self.dim_k/2), classes_info['num_output_classes'])) \
            for property_name, classes_info in self.edge_property_classes_dct.items()
        })

        self.property_type = params.property_type
        if self.property_type is not None:
            self.property_network = nn.Sequential(nn.Linear(int(self.dim_h / 2), int(self.dim_h / 4)), nn.LeakyReLU(),
                                              nn.Linear(int(self.dim_h / 4), 1))

        # for generation
        self.local_cpu = params.local_cpu
        if hasattr(params, 'num_iters'):
            self.num_iters = params.num_iters

        if len(params.graph_property_names) > 0:
            self.graph_property_embedding_layers = nn.ModuleDict({name:
                nn.Sequential(nn.Linear(1, int(self.dim_h/2)), nn.ReLU(), nn.Linear(int(self.dim_h/2), self.dim_h))
                for name in params.graph_property_names})
        else:
            self.graph_property_embedding_layers = None

        self.num_binary_graph_properties = params.num_binary_graph_properties
        if self.num_binary_graph_properties > 0:
            self.binary_graph_property_embedding_layer = nn.Linear(self.num_binary_graph_properties, 128, bias=False)

        self.predict_graph_properties = params.predict_graph_properties
        if self.predict_graph_properties is True:
            self.graph_property_predictors = nn.ModuleDict({property_name: \
                            nn.Sequential(nn.Linear(self.dim_h, 1)) for property_name in params.graph_property_names})

    def add_property_embeddings(self, node_embeddings, graph_properties, binary_graph_properties, num_nodes_per_graph):
        if self.graph_property_embedding_layers is not None:
            for name, network in self.graph_property_embedding_layers.items():
                graph_property_embedding = network(graph_properties[name])
                node_embeddings += graph_property_embedding.unsqueeze(1)
        if self.num_binary_graph_properties > 0:
            binary_graph_property_embeddings_sum = self.binary_graph_property_embedding_layer(binary_graph_properties)
            start = 0
            for i, num_nodes in enumerate(num_nodes_per_graph):
                node_embeddings[start:start+num_nodes] = node_embeddings[start:start+num_nodes] + \
                                                         binary_graph_property_embeddings_sum[i]
                start += num_nodes
        return node_embeddings

    def calculate_embeddings(self, batch_graph, graph_properties=None, binary_graph_properties=None):
        node_embeddings, edge_embeddings = [], []
        for name, property in batch_graph.ndata.items():
            node_embeddings.append(self.node_embedding_layers[name](property))
        node_embeddings = torch.stack(node_embeddings).sum(0)
        for name, property in batch_graph.edata.items():
            edge_embeddings.append(self.edge_embedding_layers[name](property))
        edge_embeddings = torch.stack(edge_embeddings).sum(0)
        node_embeddings = self.add_property_embeddings(node_embeddings, graph_properties, binary_graph_properties,
                                                       batch_graph.batch_num_nodes())
        batch_graph.ndata['nodes'] = node_embeddings
        batch_graph.edata['edge_spans'] = edge_embeddings.reshape(-1, self.dim_h, self.dim_k)

        return batch_graph

    def project_output(self, batch_init_graph, batch_scores_graph):
        batch_init_graph.edata['edge_spans'] = batch_init_graph.edata['edge_spans'].reshape(-1, self.dim_h * self.dim_k)

        for property_name, classifier in self.node_property_classifiers.items():
            batch_scores_graph.ndata[property_name] = classifier(batch_init_graph.ndata['nodes'])

        graph_property_scores = {}
        if self.predict_graph_properties is True:
            for property_name, classifier in self.graph_property_predictors.items():
                graph_property_scores[property_name] = classifier(dgl.sum_nodes(batch_init_graph))
        del batch_init_graph.ndata['nodes']

        for property_name, classifier in self.edge_property_classifiers.items():
            batch_scores_graph.edata[property_name] = classifier(batch_init_graph.edata['edge_spans'])
        del batch_init_graph.edata['edge_spans']

        return batch_init_graph, batch_scores_graph, graph_property_scores

    def predict_property(self, nodes, node_mask):
        nodes = nodes * node_mask
        pooled = nodes.mean(1)
        score = self.property_network(pooled)
        return score


class GraphNN(GraphNNParent):
    def __init__(self, params, node_loss_weights=None, edge_loss_weights=None, hydrogen_loss_weights=None):
        super().__init__(params)
        self.num_mpnns = params.num_mpnns
        mpnn = MPNN_DICT[self.mpnn_name](self.mpnn_steps, self.node_mpnn_name, self.res_conn, self.use_layer_norm,
                                         self.dim_h, self.dim_k, self.use_newest_edges, self.update_edges_at_end_only,
                                         self.fully_connected, self.spatial_msg_res_conn, self.spatial_postgru_res_conn,
                                         self.global_connection, self.bound_edges,
                                         mat_N=params.mat_N, mat_d_model=params.mat_d_model,
                                         mat_h=params.mat_h, mat_dropout=params.mat_dropout)
        self.mpnns = nn.ModuleList([copy.deepcopy(mpnn) for i in range(self.num_mpnns)])

    def forward(self, batch_init_graph, graph_properties=None, binary_graph_properties=None):
        batch_scores_graph = copy_graph_remove_data(batch_init_graph)
        batch_init_graph = self.calculate_embeddings(batch_init_graph, graph_properties, binary_graph_properties)

        for mpnn_num in range(self.num_mpnns):
            batch_init_graph = self.mpnns[mpnn_num](batch_init_graph)

        batch_init_graph, batch_scores_graph, graph_property_scores = self.project_output(batch_init_graph,
                                                                                          batch_scores_graph)

        return batch_init_graph, batch_scores_graph, graph_property_scores

class SeqGraphNN(GraphNN):
    def __init__(self, params):
        from tape import ProteinBertModel
        super().__init__(params)
        del self.node_embedding_layers
        self.seq_output_dim = params.seq_output_dim
        self.seq_model = ProteinBertModel.from_pretrained('bert-base', cache_dir='data/proteins/tape_pretrained/')
        self.seq2dim_h = nn.Linear(self.seq_output_dim, self.dim_h)

    def calculate_seq_embeddings(self, node_type, num_nodes_per_graph):
        start = 0
        padded_tape_seqs = torch.zeros(len(num_nodes_per_graph), max(num_nodes_per_graph)).long()
        masks = torch.zeros_like(padded_tape_seqs)
        for i, num_nodes in enumerate(num_nodes_per_graph):
            unpadded_deepfri_seq = node_type[start:start+num_nodes]
            unpadded_tape_seq = torch.tensor([IUPAC_VOCAB[DEEPFRI_VOCAB_KEYS[n]] for n in unpadded_deepfri_seq],
                                             dtype=unpadded_deepfri_seq.dtype)
            padded_tape_seqs[i, :num_nodes] = unpadded_tape_seq
            masks[i, :num_nodes] = 1
            start += num_nodes
        padded_tape_seqs = padded_tape_seqs.to(node_type.device)
        masks = masks.to(node_type.device)
        with torch.no_grad():
            node_embeddings = self.seq_model(padded_tape_seqs, masks)[0] # b_size, batch_max_num_nodes, seq_output_dim
        node_embeddings = node_embeddings[np.where(masks.cpu())]
        node_embeddings = self.seq2dim_h(node_embeddings)
        return node_embeddings

    def calculate_embeddings(self, batch_graph, graph_properties=None, binary_graph_properties=None):
        node_embeddings = self.calculate_seq_embeddings(batch_graph.ndata['node_type'], batch_graph.batch_num_nodes())
        edge_embeddings = []
        for name, property in batch_graph.edata.items():
            edge_embeddings.append(self.edge_embedding_layers[name](property))
        edge_embeddings = torch.stack(edge_embeddings).sum(0)
        node_embeddings = self.add_property_embeddings(node_embeddings, graph_properties, binary_graph_properties,
                                                       batch_graph.batch_num_nodes())
        batch_graph.ndata['nodes'] = node_embeddings
        batch_graph.edata['edge_spans'] = edge_embeddings.reshape(-1, self.dim_h, self.dim_k)

        return batch_graph

MODELS_DICT = {'GraphNN': GraphNN, 'SeqGraphNN': SeqGraphNN}
