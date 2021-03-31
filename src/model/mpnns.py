import copy

import dgl
import numpy as np
import torch
from torch import nn

from src.model.node_mpnns import NODE_MPNN_DICT


class MPNN(nn.Module):
    def __init__(self, mpnn_steps, node_mpnn_name, dim_h, res_conn, use_layer_norm, use_newest_edges,
                 update_edges_at_end_only, fully_connected, spatial_msg_res_conn=False,
                 spatial_postgru_res_conn=False, global_connection=False, **kwargs):
        super().__init__()
        self.mpnn_steps = mpnn_steps
        self.res_conn = res_conn
        self.node_mpnn_name = node_mpnn_name
        self.node_mpnn = NODE_MPNN_DICT[node_mpnn_name](dim_h, use_layer_norm, spatial_msg_res_conn,
                                                            spatial_postgru_res_conn, global_connection)
        self.use_newest_edges = use_newest_edges
        self.update_edges_at_end_only = update_edges_at_end_only
        self.fully_connected = fully_connected
        self.batch_graph_from_data = self.get_batch_graph_func()
        self.mpnn_step_forward = self.get_step_forward_func()

    def get_batch_graph_func(self):
        return self.batch_graph_from_data_fc if self.fully_connected is True else self.batch_graph_from_data_nonfc

    def get_step_forward_func(self):
        return self.mpnn_step_forward_fc if self.fully_connected is True else self.mpnn_step_forward_nonfc

    def batch_graph_from_data_fc(self, nodes, node_mask, adj_mat_inds):
        graph = dgl.DGLGraph()
        graph.add_nodes(nodes.shape[1])
        edge_origins = np.repeat(np.arange(nodes.shape[1]), nodes.shape[1])
        edge_dests = np.tile(np.arange(nodes.shape[1]), nodes.shape[1])
        graph.add_edges(edge_origins, edge_dests)
        graphs = dgl.batch([copy.deepcopy(graph) for b in range(nodes.shape[0])])
        return graphs

    def batch_graph_from_data_nonfc(self, nodes, node_mask, adj_mat_inds):
        graphs = []
        for j in range(nodes.shape[0]):
            graph = dgl.DGLGraph()
            graph.add_nodes(int(node_mask[j].sum()))
            graph.add_edges(*np.where(adj_mat_inds[j].cpu()))
            graphs.append(graph)
        graphs = dgl.batch(graphs)
        return graphs

    def forward(self, batch_graph):
        if self.res_conn is True:
            old_nodes, old_edges = batch_graph.ndata['nodes'], batch_graph.edata['edge_spans']
        for step_num in range(self.mpnn_steps):
            updated_nodes, updated_edges = self.mpnn_step_forward(batch_graph, step_num)
            if self.res_conn is True:
                updated_nodes = updated_nodes + old_nodes
                updated_edges = updated_edges + old_edges
            batch_graph.ndata['nodes'] = updated_nodes
            batch_graph.edata['edge_spans'] = updated_edges
        return batch_graph

    def update_nodes(self, graphs, nodes, node_mask, adj_mat_inds, edges):
        if self.node_mpnn_name == 'MAT':
            return self.node_mpnn(nodes, node_mask, adj_mat_inds, torch.zeros_like(adj_mat_inds).float(),
                                  edges.squeeze(-1).permute(0, 3, 1, 2))
        else:
            return self.node_mpnn(graphs, node_mask)

    def mpnn_step_forward_nonfc(self, batch_graph, step_num):
        if self.update_edges_at_end_only is False or step_num == self.mpnn_steps - 1:
            updated_edges = self.update_edges(batch_graph)
        else:
            updated_edges = batch_graph.edata['edge_spans']

        updated_nodes = self.node_mpnn(batch_graph)

        return updated_nodes, updated_edges

    def mpnn_step_forward_fc(self, graphs, nodes, edges, node_mask, edge_mask, adj_mat_inds, step_num):
        # calculate new node representations
        nshp, eshp = nodes.shape, edges.shape
        graphs.ndata['nodes'] = nodes.reshape(nshp[0]*nshp[1], *nshp[2:])
        graphs.edata['edge_spans'] = edges.reshape(eshp[0]*eshp[1]*eshp[2], *eshp[3:])
        new_edges = edges

        if self.update_edges_at_end_only is False or step_num == self.mpnn_steps - 1:
            # calculate new edge representations
            new_edges = self.update_edges(graphs)
            new_edges = new_edges * edge_mask.reshape(-1, 1, 1)
            if self.use_newest_edges:
                graphs.edata['edge_spans'] = new_edges
                edges_for_node_update = new_edges
            else:
                edges_for_node_update = edges

        new_nodes = self.update_nodes(graphs, nodes, node_mask, adj_mat_inds, edges_for_node_update)
        new_nodes = new_nodes.reshape(*nshp) * node_mask.unsqueeze(-1)
        new_edges = new_edges.reshape(*eshp)

        return new_nodes, new_edges

    def mpnn_step_forward_mat(self, graphs, nodes, edges, node_mask, edge_mask, adj_mat_inds, step_num):
        # calculate new node representations
        eshp = edges.shape
        graphs.edata['edge_spans'] = edges.reshape(eshp[0]*eshp[1]*eshp[2], *eshp[3:])
        new_edges = edges

        if self.update_edges_at_end_only is False or step_num == self.mpnn_steps - 1:
            # calculate new edge representations
            new_edges = self.update_edges(graphs)
            new_edges = new_edges * edge_mask.reshape(-1, 1, 1)
            if self.use_newest_edges: graphs.edata['edge_spans'] = new_edges

        new_nodes = self.node_mpnn(nodes, node_mask, adj_mat_inds, torch.zeros_like(adj_mat_inds).float(), edges)
        new_edges = new_edges.reshape(*eshp)

        return new_nodes, new_edges


class EdgesOwnRepsMPNN(MPNN):
    """MPNN where edges are updated as a function of both node and edge representations."""
    def __init__(self, mpnn_steps, node_mpnn_name, res_conn, use_layer_norm, dim_h, dim_k, use_newest_edges,
                 update_edges_at_end_only, fully_connected, spatial_msg_res_conn, spatial_postgru_res_conn,
                 global_connection, bound_edges=False, **kwargs):
        super().__init__(mpnn_steps, node_mpnn_name, dim_h, res_conn, use_layer_norm, use_newest_edges,
                         update_edges_at_end_only, fully_connected, spatial_msg_res_conn, spatial_postgru_res_conn,
                         global_connection, **kwargs)
        network_components = [nn.Linear(dim_h + dim_h * dim_k, dim_h), nn.ReLU(), nn.Linear(dim_h, dim_h * dim_k)]
        self.bound_edges = bound_edges
        if self.bound_edges is True: network_components.append(nn.Sigmoid())
        self.edge_transform_network = nn.Sequential(*network_components)

    def update_edges(self, graph):
        graph.apply_edges(dgl.function.u_add_v('nodes', 'nodes', 'msg'))
        edges = graph.edata['edge_spans']
        num_edges, dim_h, dim_k = edges.shape
        edges = edges.reshape(num_edges, dim_h*dim_k)
        edge_node_cat = torch.cat((graph.edata['msg'], edges), dim=-1)
        edges = self.edge_transform_network(edge_node_cat).reshape(num_edges, dim_h, dim_k)
        del graph.edata['msg']
        return edges


class EdgesFromNodesMPNN(MPNN):
    def __init__(self, mpnn_steps, node_mpnn_name, res_conn, use_layer_norm, dim_h, dim_k, use_newest_edges,
                 update_edges_at_end_only, fully_connected, spatial_msg_res_conn, spatial_postgru_res_conn,
                 global_connection, bound_edges=False, **kwargs):
        """MPNN where edge representations are a function of only nodes after first node mpnn update."""
        super().__init__(mpnn_steps, node_mpnn_name, dim_h, res_conn, use_layer_norm, use_newest_edges,
                         update_edges_at_end_only, fully_connected, spatial_msg_res_conn, spatial_postgru_res_conn,
                         global_connection, **kwargs)
        network_components = [nn.Linear(dim_h, int(dim_h / 2)), nn.ReLU(), nn.Linear(int(dim_h / 2), dim_h)]
        self.bound_edges = bound_edges
        if self.bound_edges is True: network_components.append(nn.Sigmoid())
        self.edge_transform_network = nn.Sequential(*network_components)

    def update_edges(self, graph):
        graph.apply_edges(dgl.function.u_add_v('nodes', 'nodes', 'msg'))
        edges = self.edge_transform_network(graph.edata['msg']).unsqueeze(-1)
        del graph.edata['msg']
        return edges

MPNN_DICT = {'EdgesFromNodesMPNN': EdgesFromNodesMPNN, 'EdgesOwnRepsMPNN': EdgesOwnRepsMPNN}