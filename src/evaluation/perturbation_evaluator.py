import json

import dgl
import numpy as np
import torch
import torch.nn.functional as F

from data.gen_targets import QM9_SYMBOL_LIST
from src.data.loader import load_graph_data
from src.model.gnn import GraphNN
from src.utils import get_index_method, get_only_target_info
from train_script_parser import get_parser


def get_final_parser():
    parser = get_parser()
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="", help='path to save evaluation output')
    parser.add_argument("--replace_hs", action='store_true', help='perturb neighbouring hydrogen as well '
                                                                'as neighbouring node')
    return parser


def main(params):
    model = GraphNN(params)
    if params.local_cpu is False: model = model.cuda()
    if params.model_path:
        sd = torch.load(params.model_path, map_location='cpu') if params.local_cpu else torch.load(params.model_path)
        model.load_state_dict(sd)
    _, _, _, val_loader = load_graph_data(params)
    preds = run_perturbations(val_loader, model, params.replace_hs, params.max_hs, params.batch_size, params.local_cpu)
    with open(params.output_path, 'w') as f:
        json.dump(preds, f)


def run_perturbations(loader, model, replace_hs, max_hs, batch_size, local_cpu):
    """
    pseudocode:
    get molecule from validation set where only one component is masked
    predict masked component
    for d in 1, ..., 4
        for j in 1, ..., number of neighbours d hops away
            corrupt neighbour j d hops away i.e. change to other random component
            predict masked component and record result
        calculate fraction of times that answer was different from original answer
    """
    preds = []
    batch_num = 0
    print(len(loader), flush=True)
    for init_nodes, init_edges, original_node_inds, _, node_masks, edge_masks, \
        node_target_types, _, init_hydrogens, _, properties in loader:
        preds.extend([{} for _ in range(init_nodes.shape[0])])
        if local_cpu is False:
            init_nodes = init_nodes.cuda()
            init_edges = init_edges.cuda()
            original_node_inds = original_node_inds.cuda()
            node_masks = node_masks.cuda()
            edge_masks = edge_masks.cuda()
            node_target_types = node_target_types.cuda()
            init_hydrogens = init_hydrogens.cuda()

        corruption_indices = np.random.randint(init_nodes.shape[1], size=init_nodes.shape[0])
        init_nodes[np.arange(init_nodes.shape[0]), corruption_indices] = loader.dataset.node_mask_index
        node_target_types[np.arange(init_nodes.shape[0]), corruption_indices] = 1
        init_hydrogens[np.arange(init_nodes.shape[0]), corruption_indices] = loader.dataset.h_mask_index
        node_target_inds_vector = getattr(node_target_types != 0, get_index_method())()

        def get_perturbed_inputs(i):
            g = dgl.DGLGraph()
            g.add_nodes(len(init_nodes[i]))
            edges = np.nonzero(init_edges[i]).T
            g.add_edges(*edges)
            all_radii_nbr_node_inds = dgl.bfs_nodes_generator(g, corruption_indices[i])
            flattened_d_hop_node_inds = torch.cat(all_radii_nbr_node_inds)
            num_nodes_per_radius = [len(r) for r in all_radii_nbr_node_inds]

            # get 3d tensor for one graph where each slice has one neighbouring node corrupted
            d_init_nodes = torch.stack([init_nodes[i]] * len(flattened_d_hop_node_inds))
            d_node_target_types = torch.stack([node_target_types[i]] * len(flattened_d_hop_node_inds))
            d_node_target_inds_vector = torch.stack([node_target_inds_vector[i]] * len(flattened_d_hop_node_inds))
            d_init_edges = torch.stack([init_edges[i]] * len(flattened_d_hop_node_inds))
            d_node_masks = torch.stack([node_masks[i]] * len(flattened_d_hop_node_inds))
            d_edge_masks = torch.stack([edge_masks[i]] * len(flattened_d_hop_node_inds))
            d_init_hydrogens = torch.stack([init_hydrogens[i]] * len(flattened_d_hop_node_inds))
            d_original_node_inds = torch.stack([original_node_inds[i]] * len(flattened_d_hop_node_inds))
            # Don't corrupt source node
            for k, j in enumerate(flattened_d_hop_node_inds[1:], 1):
                original_node = init_nodes[i][j]
                valid_choices = np.setdiff1d(np.arange(len(QM9_SYMBOL_LIST[:5])), original_node.cpu())
                replacement_node = np.random.choice(valid_choices)
                d_init_nodes[k][j] = float(replacement_node)
                if replace_hs is True:
                    original_h = init_hydrogens[i][j]
                    valid_h_choices = np.setdiff1d(np.arange(max_hs+2), original_h.cpu())
                    replacement_h = np.random.choice(valid_h_choices)
                    d_init_hydrogens[k][j] = float(replacement_h)

            return d_init_nodes, d_node_target_types, d_node_target_inds_vector, d_init_edges, d_node_masks,\
                   d_edge_masks, d_init_hydrogens, d_original_node_inds, num_nodes_per_radius

        all_inputs = [get_perturbed_inputs(i) for i in range(len(init_nodes))]
        all_init_nodes, all_node_target_types, all_node_target_inds_vector,\
            all_init_edges, all_node_masks, all_edge_masks, all_init_hydrogens,\
            all_original_node_inds = [torch.cat(inputs, dim=0) for inputs in list(zip(*all_inputs))[:-1]]

        with torch.no_grad():
            all_node_scores, _, _ = model(all_init_nodes, all_init_edges, all_node_masks,
                                                              all_edge_masks, all_init_hydrogens)
        all_node_scores, _, _ = get_only_target_info(all_node_scores, all_original_node_inds,
                                    all_node_target_inds_vector, all_node_scores.shape[-1], all_node_target_types)
        all_node_preds = torch.argmax(F.softmax(all_node_scores, -1), dim=-1).cpu().tolist()
        num_nodes_visited = 0
        for i, num_nodes_per_radius in enumerate(list(zip(*all_inputs))[-1]):
            preds[batch_num * batch_size + i] = {}
            for d, n in enumerate(num_nodes_per_radius):
                preds[batch_num*batch_size + i][d] = all_node_preds[num_nodes_visited:num_nodes_visited + n]
                num_nodes_visited += n
        batch_num += 1
        print(batch_num, flush=True)

    return preds

if __name__ == '__main__':
    parser = get_final_parser()
    params = parser.parse_args()
    main(params)
