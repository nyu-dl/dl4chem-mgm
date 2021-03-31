import dgl.function as fn
import torch
from torch import nn


class NodeMPNN(nn.Module):
    def __init__(self, dim_h, use_layer_norm, spatial_msg_res_conn=False, spatial_postgru_res_conn=False,
                 global_connection=False):
        super().__init__()
        self.dim_h = dim_h
        self.use_layer_norm = use_layer_norm
        self.gru = nn.GRU(input_size=self.dim_h, hidden_size=self.dim_h)
        if self.use_layer_norm is True:
            self.layer_norm = nn.LayerNorm(self.dim_h)
        self.spatial_msg_res_conn = spatial_msg_res_conn
        self.spatial_postgru_res_conn = spatial_postgru_res_conn
        self.global_connection = global_connection

    def update_GRU(self, msg, node):
        msg = msg.unsqueeze(0)
        node = node.unsqueeze(0)

        _, node_next = self.gru(msg, node)

        node_next = node_next.squeeze(0)
        if self.use_layer_norm is True:
            node_next = self.layer_norm(node_next)
        return node_next

    def forward(self, g, node_mask=None):
        g.send_and_recv(g.edges(), self.compute_partial_messages, self.reduce_messages)
        msg = g.ndata['messages']
        if self.global_connection is True:
            global_connection = g.ndata['nodes'].mean(dim=0)
            msg = msg + global_connection
        nodes = self.update_GRU(msg, g.ndata['nodes'])
        if self.spatial_postgru_res_conn is True:
            nodes = nodes + g.ndata['messages']
        del g.ndata['messages']
        return nodes

class MultiplicationMPNN(NodeMPNN):
    def compute_partial_messages(self, edges):
        return {'edge_spans': edges.data['edge_spans']}

    def reduce_messages(self, nodes):
        edge_spans = nodes.mailbox['edge_spans'] # shape: n, s, h, k where n = num nodes in batch, b = batch size, s=n/b
        nodes = nodes.data['nodes'].unsqueeze(-1).unsqueeze(1)  # shape: n, 1, h, 1
        messages = torch.matmul(edge_spans.permute(0, 1, 3, 2), nodes)  # shape: n, s, k, 1 where s = number of messages
        messages = torch.matmul(edge_spans, messages).squeeze(-1)  # shape: n, s, h
        messages = messages.mean(dim=1)  # shape: n, h
        return {'messages': messages}

class NbrMultMPNN(NodeMPNN):
    def compute_partial_messages(self, edges):
        partial_msg = torch.matmul(edges.data['edge_spans'].permute(0, 2, 1), edges.src['nodes'].unsqueeze(-1))
        partial_msg = torch.matmul(edges.data['edge_spans'], partial_msg).squeeze(-1)
        return {'partial_msg': partial_msg}

    def reduce_messages(self, nodes):
        partial_msg = nodes.mailbox['partial_msg']
        msg = partial_msg.mean(dim=1)
        return {'messages': msg}

class NbrEWMultMPNN(NbrMultMPNN):
    def compute_partial_messages(self, edges):
        partial_msg = edges.src['nodes'] * edges.data['edge_spans'].squeeze(-1)
        if self.spatial_msg_res_conn is True:
            partial_msg = partial_msg + edges.src['nodes']
        return {'partial_msg': partial_msg}

class AdditionMPNN(NodeMPNN):
    def compute_partial_messages(self, edges):
        return {'edge_spans': edges.data['edge_spans']}

    def reduce_messages(self, nodes):
        edge_spans = nodes.mailbox['edge_spans'] # shape: n, s, h, 1 where n = num nodes in batch, b = batch size, s=n/b
        nodes = nodes.data['nodes'].unsqueeze(-1).unsqueeze(1) # shape: n, 1, h, 1
        messages = (edge_spans + nodes).squeeze(-1) # shape: n, s, h where s = number of messages
        messages = messages.mean(dim=1) # shape: n, h
        return {'messages': messages}

class NodeAggregationMPNN(NodeMPNN):
    def compute_partial_messages(self, edges):
        return {'edge_spans': edges.data['edge_spans']}

    def reduce_messages(self, nodes):
        edge_spans = nodes.mailbox['edge_spans'] # shape: n, s, h, 1 where n = num nodes in batch, b = batch size, s=n/b
        nodes = nodes.data['nodes'].unsqueeze(-1).unsqueeze(1) # shape: n, 1, h, 1
        messages = (edge_spans + nodes).squeeze(-1) # shape: n, s, h where s = number of messages
        messages = messages.mean(dim=1) # shape: n, h
        return {'messages': messages}

class TestMPNN(NodeMPNN):
    def forward(self, g, node_mask):
        # collect features from source nodes and aggregate them in destination nodes
        g.update_all(fn.copy_src('nodes', 'message'), fn.sum('message', 'message_sum'))
        msg = g.ndata.pop('message_sum')
        nodes = self.update_GRU(msg, g.ndata['nodes'])

        g.apply_edges(fn.u_mul_v('nodes', 'nodes', 'edge_message'))
        edges = g.edata.pop('edge_spans') * g.edata.pop('edge_message').unsqueeze(-1)
        return nodes, edges

class DummyNodeMPNN(NodeMPNN):
    def forward(self, g, _):
        return g.ndata['nodes']

NODE_MPNN_DICT = {'AdditionMPNN': AdditionMPNN, 'MultiplicationMPNN': MultiplicationMPNN, 'TestMPNN': TestMPNN,
                  'NbrMultMPNN': NbrMultMPNN, 'NbrEWMultMPNN': NbrEWMultMPNN, 'DummyNodeMPNN': DummyNodeMPNN}
