import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.mpnns import MPNN_DICT
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
                                                    'num_output_classes': params.num_node_types}),
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

        self.predict_graph_properties = params.predict_graph_properties
        if self.predict_graph_properties is True:
            self.graph_property_predictors = nn.ModuleDict({property_name: \
                            nn.Sequential(nn.Linear(self.dim_h, 1)) for property_name in params.graph_property_names})

    def calculate_embeddings(self, init_node_properties, init_edge_properties, graph_properties=None):
        node_embeddings, edge_embeddings = [], []
        for name, property in init_node_properties.items():
            node_embeddings.append(self.node_embedding_layers[name](property))
        node_embeddings = torch.stack(node_embeddings).sum(0)
        for name, property in init_edge_properties.items():
            edge_embeddings.append(self.edge_embedding_layers[name](property))
        edge_embeddings = torch.stack(edge_embeddings).sum(0)
        if self.graph_property_embedding_layers is not None:
            for name, network in self.graph_property_embedding_layers.items():
                graph_property_embedding = network(graph_properties[name])
                node_embeddings += graph_property_embedding.unsqueeze(1)

        return node_embeddings, edge_embeddings.reshape(-1, self.max_nodes, self.max_nodes, self.dim_h, self.dim_k)

    def project_output(self, nodes, edge_mats):
        edges = edge_mats.reshape(-1, self.max_nodes ** 2, self.dim_h * self.dim_k)
        node_property_scores, edge_property_scores, graph_property_scores = {}, {}, {}
        for property_name, classifier in self.node_property_classifiers.items():
            node_property_scores[property_name] = classifier(nodes)
        for property_name, classifier in self.edge_property_classifiers.items():
            edge_property_scores[property_name] = classifier(edges)
            edge_property_scores[property_name] = edge_property_scores[property_name].reshape(
                -1, self.max_nodes, self.max_nodes, edge_property_scores[property_name].shape[-1])
        if self.predict_graph_properties is True:
            for property_name, classifier in self.graph_property_predictors.items():
                graph_property_scores[property_name] = classifier(nodes.sum(1))

        return node_property_scores, edge_property_scores, graph_property_scores

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

    def forward(self, init_node_properties, node_mask, init_edge_properties, edge_mask, graph_properties=None):
        self.max_nodes = init_node_properties['node_type'].shape[-1]

        nodes, edges = self.calculate_embeddings(init_node_properties, init_edge_properties, graph_properties)

        for mpnn_num in range(self.num_mpnns):
            nodes, edges = self.mpnns[mpnn_num](edges, nodes, node_mask, edge_mask, init_edge_properties['edge_type'])

        node_property_scores, edge_property_scores, graph_property_scores = self.project_output(nodes, edges)

        return node_property_scores, edge_property_scores, graph_property_scores

class GraphVAE(GraphNNParent):
    def __init__(self, params, node_loss_weights=None, edge_loss_weights=None, hydrogen_loss_weights=None):
        super().__init__(params, node_loss_weights, edge_loss_weights, hydrogen_loss_weights)
        self.num_encoder_mpnns = params.num_encoder_mpnns
        self.num_decoder_mpnns = params.num_decoder_mpnns
        mpnn = MPNN_DICT[self.mpnn_name](self.mpnn_steps, self.node_mpnn_name, self.res_conn, self.use_layer_norm,
                                         self.dim_h, self.dim_k, self.use_newest_edges, self.update_edges_at_end_only,
                                         self.fully_connected, self.spatial_msg_res_conn, self.spatial_postgru_res_conn,
                                         self.global_connection, self.bound_edges,
                                         mat_N=params.mat_N, mat_d_model=params.mat_d_model,
                                         mat_h=params.mat_h, mat_dropout=params.mat_dropout)
        self.encoder_mpnns = nn.ModuleList([copy.deepcopy(mpnn) for i in range(self.num_encoder_mpnns)])
        self.decoder_mpnns = nn.ModuleList([copy.deepcopy(mpnn) for i in range(self.num_decoder_mpnns)])
        self.node_posterior_layer = nn.Linear(self.dim_h, 2 * self.dim_h)
        self.edge_posterior_layer = nn.Linear(self.dim_h, 2 * self.dim_h * self.dim_k)

    def forward(self, node_inds, adj_mat_inds, node_mask, edge_mask, init_hydrogens):
        self.max_nodes = len(node_inds[0])

        nodes, edge_mats = self.calculate_embeddings(node_inds, adj_mat_inds, init_hydrogens)

        # Encoder MPNNs
        for mpnn_num in range(self.num_encoder_mpnns):
            nodes, edge_mats = self.encoder_mpnns[mpnn_num](edge_mats, nodes, node_mask, edge_mask, adj_mat_inds)

        # Get posterior distribution and sample latent variables
        eps = 1e-4
        nodes = self.node_posterior_layer(nodes)
        node_means, node_sds = nodes[..., :self.dim_h], F.relu(nodes[..., self.dim_h:]) + eps
        edge_mats = self.edge_posterior_layer(
            edge_mats.reshape(*edge_mats.shape[:-2], self.dim_h*self.dim_k)
            ).reshape(*edge_mats.shape[:-2], self.dim_h*2, self.dim_k)
        edge_means, edge_sds = edge_mats[..., :self.dim_h, :], F.relu(edge_mats[..., self.dim_h:, :]) + eps
        z_node = self.sample(node_means, node_sds)
        z_edge = self.sample(edge_means, edge_sds)

        # Decoder MPNNs
        for mpnn_num in range(self.num_decoder_mpnns):
            nodes, edge_mats = self.decoder_mpnns[mpnn_num](z_edge, z_node, node_mask, edge_mask, adj_mat_inds)

        node_scores, edge_scores, hydrogen_scores = self.project_output(nodes, edge_mats, init_hydrogens)
        if self.property_type is None:
            return node_scores, edge_scores, hydrogen_scores, node_means, node_sds, edge_means, edge_sds
        else:
            property_scores = self.predict_property(nodes, node_mask)
            return node_scores, edge_scores, hydrogen_scores, node_means, node_sds,\
                   edge_means, edge_sds, property_scores

    def sample(self, mean, sd):
        return mean + torch.randn_like(sd) * sd


class CGVAE(GraphNNParent):
    def __init__(self, params):
        super().__init__(params)
        self.latent_dim = int(self.dim_h/2)
        self.num_encoder_mpnns = params.num_encoder_mpnns
        self.num_decoder_mpnns = params.num_decoder_mpnns
        mpnn = MPNN_DICT[self.mpnn_name](self.mpnn_steps, self.node_mpnn_name, self.res_conn, self.use_layer_norm,
                                         self.dim_h, self.dim_k, self.use_newest_edges, self.update_edges_at_end_only,
                                         self.fully_connected, self.spatial_msg_res_conn, self.spatial_postgru_res_conn,
                                         self.global_connection, self.bound_edges,
                                         mat_N=params.mat_N, mat_d_model=params.mat_d_model,
                                         mat_h=params.mat_h, mat_dropout=params.mat_dropout)

        self.post_encoder_mpnns = nn.ModuleList([copy.deepcopy(mpnn) for i in range(self.num_encoder_mpnns)])
        self.final_node_posterior_layer = nn.Linear(self.dim_h, 2 * self.latent_dim * 6)
        self.final_edge_posterior_layer = nn.Linear(self.dim_h * self.dim_k, 2 * self.latent_dim * self.dim_k)
        self.prior_encoder_mpnns = nn.ModuleList([copy.deepcopy(mpnn) for i in range(self.num_encoder_mpnns)])
        self.final_node_prior_layer = nn.Linear(self.dim_h, 2 * self.latent_dim * 6)
        self.final_edge_prior_layer = nn.Linear(self.dim_h * self.dim_k, 2 * self.latent_dim * self.dim_k)

        self.node_decoder_fc = nn.Sequential(nn.Linear(6 * self.latent_dim + self.num_input_node_classes +
                                                       self.num_input_h_classes + self.num_input_charge_classes +
                                                       self.num_input_is_in_ring_classes +
                                                       self.num_input_is_aromatic_classes +
                                                       self.num_input_chirality_classes - 6,
                                                       self.dim_h),
                                             nn.ReLU(),
                                             nn.Linear(self.dim_h, self.dim_h))
        self.edge_decoder_fc = nn.Sequential(nn.Linear(self.latent_dim * self.dim_k + self.num_edge_types+1,
                                                       self.dim_h * self.dim_k),
                                             nn.ReLU(),
                                             nn.Linear(self.dim_h * self.dim_k, self.dim_h * self.dim_k))

        self.decoder_mpnns = nn.ModuleList([copy.deepcopy(mpnn) for i in range(self.num_decoder_mpnns)])

    def forward(self, masked_node_inds, masked_adj_mat_inds, node_mask, edge_mask, masked_hydrogens, masked_charge,
                masked_is_in_ring, masked_is_aromatic, masked_chirality,
                node_target_inds_vector, hydrogen_target_inds_vector, charge_target_inds_vector,
                is_in_ring_target_inds_vector, is_aromatic_target_inds_vector, chirality_target_inds_vector,
                edge_target_coords_matrix, unmasked_node_inds=None, unmasked_adj_mat_inds=None,
                unmasked_hydrogens=None, unmasked_charge=None, unmasked_is_in_ring=None,
                unmasked_is_aromatic=None, unmasked_chirality=None, encoder='posterior'):
        """
        b = batch size, n = max number of nodes
        :param masked_node_inds: (b, n) identity of each node with mask index for masked nodes
        :param masked_adj_mat_inds: (b, n, n) identity of each edge with mask index for masked nodes
        :param node_mask: (b, n) 1 if node exists in graph, 0 if not (zeros occur only if n > num of nodes in sample)
        :param edge_mask: (b, n, n) 1 if edge exists in graph, 0 if not. Entries for masked edges are set to 1
        :param masked_hydrogens: (b, n) number of hydrogens attached to each node with special indices for mask, etc
        :param node_target_inds_vector: (b, n) 1 where a node is masked, 0 where it is not
        :param edge_target_coords_matrix: (b, n, n) 1 where an edge is masked, 0 where it is not
        :param unmasked_node_inds: (b, n) true identity of each node (no masking)
        :param unmasked_adj_mat_inds: (b, n, n) true identity of each edge (no masking)
        :param unmasked_hydrogens: (b, n) true number of hydrogens attached to node (no masking)
        :param encoder: whether to use prior or posterior encoder
        """
        if encoder == 'posterior':
            encoder_mpnns, final_node_enc_layer, final_edge_enc_layer = self.post_encoder_mpnns, \
                                                    self.final_node_posterior_layer, self.final_edge_posterior_layer
            node_inds, adj_mat_inds, init_hydrogens, init_charge, init_is_in_ring, init_is_aromatic, init_chirality = \
                unmasked_node_inds, unmasked_adj_mat_inds,  unmasked_hydrogens, unmasked_charge, unmasked_is_in_ring, \
                unmasked_is_aromatic, unmasked_chirality
        elif encoder == 'prior':
            encoder_mpnns, final_node_enc_layer, final_edge_enc_layer = self.prior_encoder_mpnns, \
                                                    self.final_node_prior_layer, self.final_edge_prior_layer
            node_inds, adj_mat_inds, init_hydrogens, init_charge, init_is_in_ring, init_is_aromatic, init_chirality = \
                masked_node_inds, masked_adj_mat_inds, masked_hydrogens, masked_charge, masked_is_in_ring, \
                masked_is_aromatic, masked_chirality
        self.max_nodes = len(node_inds[0])

        nodes, edge_mats = self.calculate_embeddings(node_inds, adj_mat_inds, init_hydrogens, init_charge,
                                                     init_is_in_ring, init_is_aromatic, init_chirality)

        # Encoder MPNNs
        for mpnn_num in range(self.num_encoder_mpnns):
            nodes, edge_mats = encoder_mpnns[mpnn_num](edge_mats, nodes, node_mask, edge_mask, adj_mat_inds)

        # Get posterior distribution and sample latent variables
        nodes = final_node_enc_layer(nodes)
        all_node_means, all_node_logvars = nodes[..., :self.latent_dim * 6],\
                                           torch.clamp(nodes[..., self.latent_dim * 6:], -10, 10)
        node_means, hydrogen_means, charge_means, is_in_ring_means, is_aromatic_means, chirality_means = \
            all_node_means[..., :self.latent_dim], all_node_means[..., self.latent_dim:self.latent_dim * 2], \
            all_node_means[..., self.latent_dim * 2:self.latent_dim * 3], \
            all_node_means[..., self.latent_dim * 3:self.latent_dim * 4], \
            all_node_means[..., self.latent_dim * 4:self.latent_dim * 5], \
            all_node_means[..., self.latent_dim * 5:self.latent_dim * 6]
        node_logvars, hydrogen_logvars, charge_logvars, is_in_ring_logvars, is_aromatic_logvars, chirality_logvars = \
            all_node_logvars[..., :self.latent_dim], all_node_logvars[..., self.latent_dim:self.latent_dim * 2], \
            all_node_logvars[..., self.latent_dim * 2:self.latent_dim * 3], \
            all_node_logvars[..., self.latent_dim * 3:self.latent_dim * 4], \
            all_node_logvars[..., self.latent_dim * 4:self.latent_dim * 5], \
            all_node_logvars[..., self.latent_dim * 5:self.latent_dim * 6]
        edge_mats = final_edge_enc_layer(
            edge_mats.reshape(*edge_mats.shape[:-2], self.dim_h*self.dim_k)
            ).reshape(*edge_mats.shape[:-2], 2*self.latent_dim, self.dim_k)
        edge_means, edge_logvars = edge_mats[..., :self.latent_dim, :], torch.clamp(edge_mats[..., self.latent_dim:, :], -10, 10)

        def property_posterior_to_dec_embedding(means, logvars, target_inds_vector, masked_property, num_input_classes):
            z_property = self.sample(means, logvars)
            z_property[target_inds_vector] = 0
            property_onehot = F.one_hot(masked_property, num_input_classes)[..., :num_input_classes-1].float()
            # Concatenate conditioning variables
            z_property_cond = torch.cat((z_property, property_onehot.to(z_property.device)), dim=-1)

            return z_property_cond

        z_node_cond = property_posterior_to_dec_embedding(node_means, node_logvars, node_target_inds_vector,
                                                     masked_node_inds, self.num_input_node_classes)
        z_hydrogen_cond = property_posterior_to_dec_embedding(hydrogen_means, hydrogen_logvars,
                                                hydrogen_target_inds_vector, masked_hydrogens, self.num_input_h_classes)
        z_charge_cond = property_posterior_to_dec_embedding(charge_means, charge_logvars, charge_target_inds_vector,
                                                            masked_charge, self.num_input_charge_classes)
        z_is_in_ring_cond = property_posterior_to_dec_embedding(is_in_ring_means, is_in_ring_logvars,
                                    is_in_ring_target_inds_vector, masked_is_in_ring, self.num_input_is_in_ring_classes)
        z_is_aromatic_cond = property_posterior_to_dec_embedding(is_aromatic_means, is_aromatic_logvars,
                                is_aromatic_target_inds_vector, masked_is_aromatic, self.num_input_is_aromatic_classes)
        z_chirality_cond = property_posterior_to_dec_embedding(chirality_means, chirality_logvars,
                                    chirality_target_inds_vector, masked_chirality, self.num_input_chirality_classes)

        z_all_node_properties_cond = torch.cat((z_node_cond, z_hydrogen_cond, z_charge_cond, z_is_in_ring_cond,
                                                z_is_aromatic_cond, z_chirality_cond), dim=-1)

        z_edge = self.sample(edge_means, edge_logvars)
        z_edge[(edge_target_coords_matrix + torch.transpose(edge_target_coords_matrix, 1, 2)) == 0] = 0

        edge_onehot = F.one_hot(masked_adj_mat_inds, self.num_input_edge_classes).float()
        edge_onehot = edge_onehot[..., :self.num_edge_types+1]
        z_edge_cond = torch.cat((z_edge.reshape(*z_edge.shape[:-2], z_edge.shape[-2]*z_edge.shape[-1]),
                                 edge_onehot.to(z_edge.device)), dim=-1)

        # Fully connected layers
        z_consolidated_node_cond = self.node_decoder_fc(z_all_node_properties_cond) * node_mask.unsqueeze(-1)
        z_edge_cond = self.edge_decoder_fc(z_edge_cond) * edge_mask.unsqueeze(-1)
        z_edge_cond = z_edge_cond.reshape(*z_edge_cond.shape[:-1], self.dim_h, self.dim_k)

        # Decoder MPNNs
        for mpnn_num in range(self.num_decoder_mpnns):
            nodes, edge_mats = self.decoder_mpnns[mpnn_num](z_edge_cond, z_consolidated_node_cond, node_mask, edge_mask,
                                                            masked_adj_mat_inds)

        node_scores, edge_scores, hydrogen_scores, charge_scores, is_in_ring_scores, is_aromatic_scores, \
                chirality_scores = self.project_output(nodes, edge_mats, init_hydrogens)

        return (node_scores, hydrogen_scores, charge_scores, is_in_ring_scores, is_aromatic_scores,
                chirality_scores), edge_scores, (node_means, node_logvars, hydrogen_means, hydrogen_logvars, charge_means,
                charge_logvars, is_in_ring_means, is_in_ring_logvars, is_aromatic_means, is_aromatic_logvars,
                chirality_means, chirality_logvars), (edge_means, edge_logvars)

    def prior_forward(self, masked_nodes, masked_edges, masked_hydrogens, masked_charge, masked_is_in_ring,
                      masked_is_aromatic, masked_chirality, node_mask, edge_mask,
                      node_target_inds_vector, hydrogen_target_inds_vector, charge_target_inds_vector,
                      is_in_ring_target_inds_vector, is_aromatic_target_inds_vector, chirality_target_inds_vector,
                      edge_target_coords_matrix):
        return self(masked_nodes, masked_edges, node_mask, edge_mask, masked_hydrogens, masked_charge,
                    masked_is_in_ring, masked_is_aromatic, masked_chirality,
                    node_target_inds_vector, hydrogen_target_inds_vector, charge_target_inds_vector,
                    is_in_ring_target_inds_vector, is_aromatic_target_inds_vector, chirality_target_inds_vector,
                    edge_target_coords_matrix, encoder='prior')

    def posterior_forward(self, masked_nodes, masked_edges, masked_hydrogens, masked_charge, masked_is_in_ring,
                          masked_is_aromatic, masked_chirality, unmasked_nodes, unmasked_edges, unmasked_hydrogens,
                          unmasked_charge, unmasked_is_in_ring, unmasked_is_aromatic, unmasked_chirality,
                          node_mask, edge_mask, node_target_inds_vector, hydrogen_target_inds_vector,
                          charge_target_inds_vector, is_in_ring_target_inds_vector, is_aromatic_target_inds_vector,
                          chirality_target_inds_vector, edge_target_coords_matrix):
        return self(masked_nodes, masked_edges, node_mask, edge_mask, masked_hydrogens, masked_charge,
                    masked_is_in_ring, masked_is_aromatic, masked_chirality,
                    node_target_inds_vector, hydrogen_target_inds_vector, charge_target_inds_vector,
                    is_in_ring_target_inds_vector, is_aromatic_target_inds_vector, chirality_target_inds_vector,
                    edge_target_coords_matrix, unmasked_nodes, unmasked_edges,
                    unmasked_hydrogens, unmasked_charge, unmasked_is_in_ring, unmasked_is_aromatic, unmasked_chirality,
                    encoder='posterior')

    def sample(self, mean, logvar):
        return mean + torch.randn_like(logvar) * torch.exp(0.5*logvar)


MODELS_DICT = {'GraphNN': GraphNN, 'GraphVAE': GraphVAE, 'CGVAE': CGVAE}
