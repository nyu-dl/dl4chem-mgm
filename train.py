import os
import pprint
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from src.data.loader import load_graph_data
from src.evaluation.nbrhood_stats_aggregator import aggregate_stats
from src.evaluation.perturbation_evaluator import run_perturbations
from src.logger import create_logger
from src.model.gnn import MODELS_DICT
from src.model.graph_generator import GraphGenerator
from src.utils import set_seed_if, get_optimizer, get_index_method, get_only_target_info, \
    get_ds_stats, check_validity, write_tensorboard, get_all_result_distributions, update_ds_stats, \
    accuracies_from_totals, lr_decay_multiplier, save_checkpoints, calculate_gen_benchmarks, get_loss_weights, get_loss, \
    get_majority_and_minority_stats
from train_script_parser import get_parser


def setup_data_and_model(params, model):
    # Variables that may not otherwise be assigned
    writer = perturbation_loader = generator = training_smiles = None

    # setup random seeds
    if params.val_seed is None: params.val_seed = params.seed
    set_seed_if(params.seed)

    exp_path = os.path.join(params.dump_path, params.exp_name)
    # create exp path if it doesn't exist
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    # create logger
    logger = create_logger(os.path.join(exp_path, 'train.log'), 0)
    pp = pprint.PrettyPrinter()
    logger.info("============ Initialized logger ============")
    logger.info("Random seed is {}".format(params.seed))
    if params.suppress_params is False:
        logger.info("\n".join("%s: %s" % (k, str(v))
                          for k, v in sorted(dict(vars(params)).items())))
        logger.info("Running command: %s" % 'python ' + ' '.join(sys.argv))
    logger.info("The experiment will be stored in %s\n" % exp_path)
    logger.info("")
    # load data
    train_data, val_dataset, train_loader, val_loader = load_graph_data(params)
    if params.run_perturbation_analysis is True:
        params.do_not_corrupt = True
        params.batch_size = params.perturbation_batch_size
        params.edges_per_batch = params.perturbation_edges_per_batch
        _, _, _, perturbation_loader = load_graph_data(params)
        del _

    logger.info ('train_loader len is {}'.format(len(train_loader)))
    logger.info ('val_loader len is {}'.format(len(val_loader)))

    if params.load_latest is True:
        load_prefix = 'latest'
    elif params.load_best is True:
        load_prefix = 'best'
    else:
        load_prefix = None

    if load_prefix is not None:
        if params.local_cpu is True:
            model.load_state_dict(torch.load(os.path.join(exp_path, '{}_model'.format(load_prefix)), map_location='cpu'))
        else:
            model.load_state_dict(torch.load(os.path.join(exp_path, '{}_model'.format(load_prefix))))
    if params.local_cpu is False:
        model = model.cuda()
    if params.gen_num_samples > 0:
        generator = GraphGenerator(train_data, model, params.gen_random_init, params.gen_num_iters, params.gen_predict_deterministically, params.local_cpu)
        with open(params.smiles_path) as f:
            smiles = f.read().split('\n')
            training_smiles = smiles[:int(params.smiles_train_split * len(smiles))]
            del smiles
    opt = get_optimizer(model.parameters(), params.optimizer)
    if load_prefix is not None:
        opt.load_state_dict(torch.load(os.path.join(exp_path, '{}_opt_sd'.format(load_prefix))))

    lr = opt.param_groups[0]['lr']
    lr_lambda = lambda iteration: lr_decay_multiplier(iteration, params.warm_up_iters, params.decay_start_iter,
                                                      params.lr_decay_amount, params.lr_decay_frac,
                                                      params.lr_decay_interval, params.min_lr, lr)
    scheduler = LambdaLR(opt, lr_lambda)
    index_method = get_index_method()

    best_loss = 9999
    if params.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(exp_path)

    total_iter, grad_accum_iters = params.first_iter, 0

    return params, model, opt, scheduler, train_data, train_loader, val_dataset, val_loader, perturbation_loader,\
           generator, index_method, exp_path, training_smiles, pp, logger, writer, best_loss, total_iter,\
           grad_accum_iters


def main(params):
    model_cls = MODELS_DICT[params.model_name]
    model = model_cls(params)
    params, model, opt, scheduler, train_data, train_loader, val_dataset, val_loader, perturbation_loader, generator,\
    index_method, exp_path, training_smiles, pp, logger, writer, best_loss,\
    total_iter, grad_accum_iters = setup_data_and_model(params, model)

    for epoch in range(1, params.num_epochs+1):
        print('Starting epoch {}'.format(epoch), flush=True)
        for train_batch in train_loader:
            if total_iter % 100 == 0: print(total_iter, flush=True)
            if total_iter == params.max_steps:
                logger.info('Done training')
                break
            model.train()

            # Training step
            init_nodes, init_edges, original_node_inds, original_adj_mats, node_masks, edge_masks,\
            node_target_types, edge_target_types, init_hydrogens, original_hydrogens, init_charge,\
            orig_charge, init_is_in_ring, orig_is_in_ring, init_is_aromatic, orig_is_aromatic, init_chirality,\
            orig_chirality, hydrogen_target_types, charge_target_types, is_in_ring_target_types,\
            is_aromatic_target_types, chirality_target_types = train_batch
            # init is what goes into model
            # target_inds and target_coords are 1 at locations to be predicted, 0 elsewhere
            # target_inds and target_coords are now calculated here rather than in dataloader
            # original are uncorrupted data (used for comparison with prediction in loss calculation)
            # masks are 1 in places corresponding to nodes that exist, 0 in other places (which are empty/padded)
            # target_types are 1 in places to mask, 2 in places to replace, 3 in places to reconstruct, 0 in places not to predict

            node_target_inds_vector = getattr(node_target_types != 0, index_method)()
            edge_target_coords_matrix = getattr(edge_target_types != 0, index_method)()
            hydrogen_target_inds_vector = getattr(hydrogen_target_types != 0, index_method)()
            charge_target_inds_vector = getattr(charge_target_types != 0, index_method)()
            is_in_ring_target_inds_vector = getattr(is_in_ring_target_types != 0, index_method)()
            is_aromatic_target_inds_vector = getattr(is_aromatic_target_types != 0, index_method)()
            chirality_target_inds_vector = getattr(chirality_target_types != 0, index_method)()

            if params.local_cpu is False:
                init_nodes = init_nodes.cuda()
                init_edges = init_edges.cuda()
                original_node_inds = original_node_inds.cuda()
                original_adj_mats = original_adj_mats.cuda()
                node_masks = node_masks.cuda()
                edge_masks = edge_masks.cuda()
                node_target_types = node_target_types.cuda()
                edge_target_types = edge_target_types.cuda()
                init_hydrogens = init_hydrogens.cuda()
                original_hydrogens = original_hydrogens.cuda()
                init_charge = init_charge.cuda()
                orig_charge = orig_charge.cuda()
                init_is_in_ring = init_is_in_ring.cuda()
                orig_is_in_ring = orig_is_in_ring.cuda()
                init_is_aromatic = init_is_aromatic.cuda()
                orig_is_aromatic = orig_is_aromatic.cuda()
                init_chirality = init_chirality.cuda()
                orig_chirality = orig_chirality.cuda()
                hydrogen_target_types = hydrogen_target_types.cuda()
                charge_target_types = charge_target_types.cuda()
                is_in_ring_target_types = is_in_ring_target_types.cuda()
                is_aromatic_target_types = is_aromatic_target_types.cuda()
                chirality_target_types = chirality_target_types.cuda()
                if params.property_type is not None: properties = properties.cuda()

            if grad_accum_iters % params.grad_accum_iters == 0:
                opt.zero_grad()

            out = model(init_nodes, init_edges, node_masks, edge_masks, init_hydrogens, init_charge, init_is_in_ring,
                        init_is_aromatic, init_chirality)
            node_scores, edge_scores, hydrogen_scores, charge_scores, is_in_ring_scores, is_aromatic_scores,\
                chirality_scores = out
            node_num_classes = node_scores.shape[-1]
            edge_num_classes = edge_scores.shape[-1]
            hydrogen_num_classes = hydrogen_scores.shape[-1]
            charge_num_classes = charge_scores.shape[-1]
            is_in_ring_num_classes = is_in_ring_scores.shape[-1]
            is_aromatic_num_classes = is_aromatic_scores.shape[-1]
            chirality_num_classes = chirality_scores.shape[-1]

            if model.property_type is not None:
                property_scores = out[-1]

            # slice out target data structures
            node_scores, target_nodes, node_target_types = get_only_target_info(node_scores, original_node_inds,
                                                node_target_inds_vector, node_num_classes, node_target_types)
            edge_scores, target_adj_mats, edge_target_types = get_only_target_info(edge_scores, original_adj_mats,
                                                edge_target_coords_matrix, edge_num_classes, edge_target_types)
            if params.embed_hs is True:
                hydrogen_scores, target_hydrogens, hydrogen_target_types = get_only_target_info(hydrogen_scores,
                        original_hydrogens, hydrogen_target_inds_vector, hydrogen_num_classes, hydrogen_target_types)
                charge_scores, target_charge, charge_target_types = get_only_target_info(charge_scores, orig_charge,
                                        charge_target_inds_vector, charge_num_classes, charge_target_types)
                is_in_ring_scores, target_is_in_ring, is_in_ring_target_types = get_only_target_info(is_in_ring_scores,
                        orig_is_in_ring, is_in_ring_target_inds_vector, is_in_ring_num_classes, is_in_ring_target_types)
                is_aromatic_scores, target_is_aromatic, is_aromatic_target_types = get_only_target_info(
                                    is_aromatic_scores, orig_is_aromatic,
                                    is_aromatic_target_inds_vector, is_aromatic_num_classes, is_aromatic_target_types)
                chirality_scores, target_chirality, chirality_target_types = get_only_target_info(chirality_scores,
                            orig_chirality, chirality_target_inds_vector, chirality_num_classes, chirality_target_types)


            num_components = len(target_nodes) + len(target_adj_mats)
            if params.embed_hs is True: num_components += len(target_hydrogens)
            # calculate score
            losses = []
            results_dict = {}
            metrics_to_print = []
            if params.target_data_structs in ['nodes', 'both', 'random'] and len(target_nodes) > 0:
                node_loss = get_loss(target_nodes, node_scores, params.equalise, params.loss_normalisation_type,
                                     num_components, params.local_cpu)
                losses.append(node_loss)
                node_preds = torch.argmax(F.softmax(node_scores, -1), dim=-1)
                nodes_acc, mask_node_acc, replace_node_acc, recon_node_acc = get_ds_stats(node_scores, target_nodes,
                                                                                          node_target_types)
                carbon_nodes_correct, noncarbon_nodes_correct, carbon_nodes_acc, noncarbon_nodes_acc = \
                    get_majority_and_minority_stats(node_preds, target_nodes, 1)
                results_dict.update({'nodes_acc': nodes_acc, 'carbon_nodes_acc': carbon_nodes_acc,
                    'noncarbon_nodes_acc': noncarbon_nodes_acc, 'mask_node_acc': mask_node_acc,
                    'replace_node_acc': replace_node_acc, 'recon_node_acc': recon_node_acc, 'node_loss': node_loss})
                metrics_to_print.extend(['node_loss', 'nodes_acc', 'carbon_nodes_acc', 'noncarbon_nodes_acc'])

                def node_property_computations(name, scores, targets, target_types, binary=False):
                    loss = get_loss(targets, scores, params.equalise, params.loss_normalisation_type, num_components,
                                    params.local_cpu, binary=binary)
                    losses.append(loss)
                    acc, mask_acc, replace_acc, recon_acc = get_ds_stats(scores, targets, target_types)
                    results_dict.update({'{}_acc'.format(name): acc, 'mask_{}_acc'.format(name): mask_acc,
                                    'replace_{}_acc'.format(name): replace_acc, 'recon_{}_acc'.format(name): recon_acc,
                                    '{}_loss'.format(name): loss})
                    metrics_to_print.extend(['{}_loss'.format(name)])

                if params.embed_hs is True:
                    node_property_computations('hydrogen', hydrogen_scores, target_hydrogens, hydrogen_target_types)
                    node_property_computations('charge', charge_scores, target_charge, charge_target_types)
                    node_property_computations('is_in_ring', is_in_ring_scores, target_is_in_ring,
                                               is_in_ring_target_types)
                    node_property_computations('is_aromatic', is_aromatic_scores, target_is_aromatic,
                                               is_aromatic_target_types)
                    node_property_computations('chirality', chirality_scores, target_chirality, chirality_target_types)

            if params.target_data_structs in ['edges', 'both', 'random'] and len(target_adj_mats) > 0:
                edge_loss = get_loss(target_adj_mats, edge_scores, params.equalise,
                                         params.loss_normalisation_type, num_components, params.local_cpu)
                losses.append(edge_loss)
                edge_preds = torch.argmax(F.softmax(edge_scores, -1), dim=-1)
                edges_acc, mask_edge_acc, replace_edge_acc, recon_edge_acc = get_ds_stats(edge_scores, target_adj_mats,
                                                                                          edge_target_types)
                no_edge_correct, edge_present_correct, no_edge_acc, edge_present_acc = \
                    get_majority_and_minority_stats(edge_preds, target_adj_mats, 0)
                results_dict.update({'edges_acc': edges_acc, 'edge_present_acc': edge_present_acc,
                        'no_edge_acc': no_edge_acc, 'mask_edge_acc': mask_edge_acc, 'replace_edge_acc': replace_edge_acc,
                        'recon_edge_acc': recon_edge_acc, 'edge_loss': edge_loss})
                metrics_to_print.extend(['edge_loss', 'edges_acc', 'edge_present_acc', 'no_edge_acc'])

            if params.property_type is not None:
                property_loss = model.property_loss(property_scores, properties)/params.batch_size
                losses.append(property_loss)
                results_dict.update({'property_loss': property_loss})
                metrics_to_print.extend(['property_loss'])

            loss = sum(losses)
            if params.no_update is False:
                (loss/params.grad_accum_iters).backward()
                grad_accum_iters += 1
                if grad_accum_iters % params.grad_accum_iters == 0:
                    # clip grad norm
                    if params.clip_grad_norm > -1:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), params.clip_grad_norm)
                    opt.step()
                    grad_accum_iters = 0
                if (total_iter+1) <= params.warm_up_iters or (total_iter+1) % params.lr_decay_interval == 0:
                    scheduler.step(total_iter+1)

            if params.check_pred_validity is True:
                if params.target_data_structs in ['nodes', 'both', 'random'] and len(target_nodes) > 0:
                    init_nodes[node_target_inds_vector] = node_preds
                if params.target_data_structs in ['edges', 'both', 'random'] and len(target_adj_mats) > 0:
                    init_edges[edge_target_coords_matrix] = edge_preds
                is_valid_list, is_connected_list = check_validity(init_nodes, init_edges)
                percent_valid = np.mean(is_valid_list) * 100
                valid_percent_connected = np.mean(is_connected_list) * 100
                results_dict.update({'percent_valid': percent_valid, 'percent_connected': valid_percent_connected})

            if params.suppress_train_log is False:
                log_string = ''
                for name, value in results_dict.items():
                    if name in metrics_to_print:
                        log_string += ', {} = {:.2f}'.format(name, value)
                log_string = 'total_iter = {0:d}, loss = {1:.2f}'.format(total_iter, loss.cpu().item()) + log_string
                logger.info(log_string)

            if params.target_frac_inc_after is not None and total_iter > 0 and total_iter % params.target_frac_inc_after == 0:
                train_data.node_target_frac = min(train_data.node_target_frac + params.target_frac_inc_amount,
                                                  params.max_target_frac)
                train_data.edge_target_frac = min(train_data.edge_target_frac + params.target_frac_inc_amount,
                                                  params.max_target_frac)
            results_dict.update({'node_target_frac': train_data.node_target_frac})
            results_dict.update({'edge_target_frac': train_data.edge_target_frac})

            if params.tensorboard and total_iter % int(params.log_train_steps) == 0:
                results_dict.update({'loss': loss, 'lr': opt.param_groups[0]['lr']})
                write_tensorboard(writer, 'train', results_dict, total_iter)

            dist_names = [
            ['mask_edge_correct_dist', 'mask_edge_incorrect_pred_dist', 'mask_edge_incorrect_true_dist'],
            ['replace_edge_correct_dist', 'replace_edge_incorrect_pred_dist', 'replace_edge_incorrect_true_dist'],
            ['recon_edge_correct_dist', 'recon_edge_incorrect_pred_dist', 'recon_edge_incorrect_true_dist']
            ]
            distributions = np.zeros((3, 3, params.num_edge_types))
            if total_iter > 0 and total_iter % params.val_after == 0:
                logger.info('Validating')
                val_loss, property_loss, node_loss, edge_loss, hydrogen_loss, num_data_points = 0, 0, 0, 0, 0, 0
                charge_loss, is_in_ring_loss, is_aromatic_loss, chirality_loss = 0, 0, 0, 0
                model.eval()
                set_seed_if(params.seed)
                nodes_correct, edges_correct, total_nodes, total_edges = 0, 0, 0, 0
                carbon_nodes_correct, noncarbon_nodes_correct, total_carbon_nodes, total_noncarbon_nodes = 0, 0, 0, 0
                mask_nodes_correct, replace_nodes_correct, recon_nodes_correct = 0, 0, 0
                total_mask_nodes, total_replace_nodes, total_recon_nodes = 0, 0, 0
                no_edge_correct, edge_present_correct, total_no_edges, total_edges_present = 0, 0, 0, 0
                mask_edges_correct, replace_edges_correct, recon_edges_correct = 0, 0, 0
                total_mask_edges, total_replace_edges, total_recon_edges = 0, 0, 0
                hydrogens_correct, mask_hydrogens_correct, replace_hydrogens_correct, recon_hydrogens_correct, \
                    total_mask_hydrogens, total_replace_hydrogens, total_recon_hydrogens,\
                    total_hydrogens = 0, 0, 0, 0, 0, 0, 0, 0
                is_valid_list, is_connected_list = [], []
                for init_nodes, init_edges, original_node_inds, original_adj_mats, node_masks, edge_masks,\
                    node_target_types, edge_target_types, init_hydrogens, original_hydrogens,\
                    init_charge, orig_charge, init_is_in_ring, orig_is_in_ring, init_is_aromatic, orig_is_aromatic,\
                    init_chirality, orig_chirality, hydrogen_target_types, charge_target_types, is_in_ring_target_types,\
                    is_aromatic_target_types, chirality_target_types in val_loader:

                    node_target_inds_vector = getattr(node_target_types != 0, index_method)()
                    edge_target_coords_matrix = getattr(edge_target_types != 0, index_method)()
                    hydrogen_target_inds_vector = getattr(hydrogen_target_types != 0, index_method)()
                    charge_target_inds_vector = getattr(charge_target_types != 0, index_method)()
                    is_in_ring_target_inds_vector = getattr(is_in_ring_target_types != 0, index_method)()
                    is_aromatic_target_inds_vector = getattr(is_aromatic_target_types != 0, index_method)()
                    chirality_target_inds_vector = getattr(chirality_target_types != 0, index_method)()

                    if params.local_cpu is False:
                        init_nodes = init_nodes.cuda()
                        init_edges = init_edges.cuda()
                        original_node_inds = original_node_inds.cuda()
                        original_adj_mats = original_adj_mats.cuda()
                        node_masks = node_masks.cuda()
                        edge_masks = edge_masks.cuda()
                        node_target_types = node_target_types.cuda()
                        edge_target_types = edge_target_types.cuda()
                        init_hydrogens = init_hydrogens.cuda()
                        original_hydrogens = original_hydrogens.cuda()
                        init_charge = init_charge.cuda()
                        orig_charge = orig_charge.cuda()
                        init_is_in_ring = init_is_in_ring.cuda()
                        orig_is_in_ring = orig_is_in_ring.cuda()
                        init_is_aromatic = init_is_aromatic.cuda()
                        orig_is_aromatic = orig_is_aromatic.cuda()
                        init_chirality = init_chirality.cuda()
                        orig_chirality = orig_chirality.cuda()
                        hydrogen_target_types = hydrogen_target_types.cuda()
                        charge_target_types = charge_target_types.cuda()
                        is_in_ring_target_types = is_in_ring_target_types.cuda()
                        is_aromatic_target_types = is_aromatic_target_types.cuda()
                        chirality_target_types = chirality_target_types.cuda()
                        if params.embed_hs is True:
                            original_hydrogens = original_hydrogens.cuda()
                        if params.property_type is not None: properties = properties.cuda()

                    batch_size = init_nodes.shape[0]

                    with torch.no_grad():
                        out = model(init_nodes, init_edges, node_masks, edge_masks, init_hydrogens, init_charge,
                                    init_is_in_ring, init_is_aromatic, init_chirality)
                        node_scores, edge_scores, hydrogen_scores, charge_scores, is_in_ring_scores,\
                        is_aromatic_scores, chirality_scores = out
                        if model.property_type is not None:
                            property_scores = out[-1]

                    node_num_classes = node_scores.shape[-1]
                    edge_num_classes = edge_scores.shape[-1]
                    hydrogen_num_classes = hydrogen_scores.shape[-1]
                    charge_num_classes = charge_scores.shape[-1]
                    is_in_ring_num_classes = is_in_ring_scores.shape[-1]
                    is_aromatic_num_classes = is_aromatic_scores.shape[-1]
                    chirality_num_classes = chirality_scores.shape[-1]

                    node_scores, target_nodes, node_target_types = get_only_target_info(node_scores, original_node_inds,
                                                node_target_inds_vector, node_num_classes, node_target_types)
                    edge_scores, target_adj_mats, edge_target_types = get_only_target_info(edge_scores,
                            original_adj_mats, edge_target_coords_matrix, edge_num_classes, edge_target_types)

                    if params.embed_hs is True:
                        hydrogen_scores, target_hydrogens, hydrogen_target_types = get_only_target_info(hydrogen_scores,
                                                                                    original_hydrogens,
                                                                                    hydrogen_target_inds_vector,
                                                                                    hydrogen_num_classes,
                                                                                    hydrogen_target_types)
                        charge_scores, target_charge, charge_target_types = get_only_target_info(charge_scores,
                                                                               orig_charge,
                                                                               charge_target_inds_vector,
                                                                               charge_num_classes, charge_target_types)
                        is_in_ring_scores, target_is_in_ring, is_in_ring_target_types = get_only_target_info(
                                                                                       is_in_ring_scores,
                                                                                       orig_is_in_ring,
                                                                                       is_in_ring_target_inds_vector,
                                                                                       is_in_ring_num_classes,
                                                                                       is_in_ring_target_types)
                        is_aromatic_scores, target_is_aromatic, is_aromatic_target_types = get_only_target_info(
                                                                                         is_aromatic_scores,
                                                                                         orig_is_aromatic,
                                                                                         is_aromatic_target_inds_vector,
                                                                                         is_aromatic_num_classes,
                                                                                         is_aromatic_target_types)
                        chirality_scores, target_chirality, chirality_target_types = get_only_target_info(
                                                                                     chirality_scores, orig_chirality,
                                                                                     chirality_target_inds_vector,
                                                                                     chirality_num_classes,
                                                                                     chirality_target_types)
                    num_data_points += batch_size

                    losses = []
                    if params.target_data_structs in ['nodes', 'both', 'random'] and len(target_nodes) > 0:
                        weight = get_loss_weights(target_nodes, node_scores, params.equalise, params.local_cpu)
                        iter_node_loss = F.cross_entropy(node_scores, target_nodes, weight=weight, reduction='sum')
                        losses.append(iter_node_loss)
                        node_loss += iter_node_loss
                        nodes_correct, mask_nodes_correct, replace_nodes_correct, recon_nodes_correct, total_mask_nodes, \
                        total_replace_nodes, total_recon_nodes, total_nodes = update_ds_stats(node_scores, target_nodes,
                            node_target_types, nodes_correct, mask_nodes_correct, replace_nodes_correct,
                            recon_nodes_correct, total_mask_nodes, total_replace_nodes, total_recon_nodes, total_nodes)
                        total_noncarbon_nodes += (target_nodes != 1).sum()
                        total_carbon_nodes += (target_nodes == 1).sum()
                        node_preds = torch.argmax(F.softmax(node_scores, -1), dim=-1)
                        noncarbon_nodes_correct += torch.mul((node_preds == target_nodes), (target_nodes != 1)).sum()
                        carbon_nodes_correct += torch.mul((node_preds == target_nodes), (target_nodes == 1)).sum()
                        if params.check_pred_validity is True:
                            init_nodes[node_target_inds_vector] = node_preds

                        def val_node_property_loss_computation(targets, scores, loss, binary=False):
                            weight = get_loss_weights(targets, scores, params.equalise, params.local_cpu)
                            if binary is True:
                                iter_loss = F.binary_cross_entropy_with_logits(scores, targets, weight=weight,
                                                                               reduction='sum')
                            else:
                                iter_loss = F.cross_entropy(scores, targets, weight=weight, reduction='sum')
                            losses.append(iter_loss)
                            loss += iter_loss
                            return loss

                        if params.embed_hs is True:
                            hydrogen_loss = val_node_property_loss_computation(target_hydrogens, hydrogen_scores,
                                                                               hydrogen_loss)
                            hydrogens_correct, mask_hydrogens_correct, replace_hydrogens_correct, recon_hydrogens_correct,\
                            total_mask_hydrogens, total_replace_hydrogens, total_recon_hydrogens, total_hydrogens =\
                                update_ds_stats(hydrogen_scores, target_hydrogens, hydrogen_target_types, hydrogens_correct,
                                mask_hydrogens_correct, replace_hydrogens_correct, recon_hydrogens_correct,
                                total_mask_hydrogens, total_replace_hydrogens, total_recon_hydrogens, total_hydrogens)
                            charge_loss = val_node_property_loss_computation(target_charge, charge_scores,
                                                                               charge_loss)
                            is_in_ring_loss = val_node_property_loss_computation(target_is_in_ring, is_in_ring_scores,
                                                                               is_in_ring_loss)
                            is_aromatic_loss = val_node_property_loss_computation(target_is_aromatic,
                                                                    is_aromatic_scores, is_aromatic_loss)
                            chirality_loss = val_node_property_loss_computation(target_chirality, chirality_scores,
                                                                               chirality_loss)

                    if params.target_data_structs in ['edges', 'both', 'random'] and len(target_adj_mats) > 0:
                        weight = get_loss_weights(target_adj_mats, edge_scores, params.equalise, params.local_cpu)
                        iter_edge_loss = F.cross_entropy(edge_scores, target_adj_mats, weight=weight, reduction='sum')
                        losses.append(iter_edge_loss)
                        edge_loss += iter_edge_loss
                        edges_correct, mask_edges_correct, replace_edges_correct, recon_edges_correct, total_mask_edges, \
                        total_replace_edges, total_recon_edges, total_edges = update_ds_stats(edge_scores, target_adj_mats,
                            edge_target_types, edges_correct, mask_edges_correct, replace_edges_correct,
                            recon_edges_correct, total_mask_edges, total_replace_edges, total_recon_edges, total_edges)
                        total_edges_present += (target_adj_mats != 0).sum()
                        total_no_edges += (target_adj_mats == 0).sum()
                        edge_preds = torch.argmax(F.softmax(edge_scores, -1), dim=-1)
                        edge_present_correct += torch.mul((edge_preds == target_adj_mats), (target_adj_mats != 0)).sum()
                        no_edge_correct += torch.mul((edge_preds == target_adj_mats), (target_adj_mats == 0)).sum()

                        distributions += get_all_result_distributions(edge_preds, target_adj_mats, edge_target_types, [1, 2, 3],
                                                                      params.num_edge_types)
                        if params.check_pred_validity is True:
                            init_edges[edge_target_coords_matrix] = edge_preds

                    if params.property_type is not None:
                        iter_property_loss = model.property_loss(property_scores, properties)
                        losses.append(iter_property_loss)
                        property_loss += iter_property_loss

                    loss = sum(losses).cpu().item()
                    val_loss += loss

                    if params.check_pred_validity is True:
                        is_valid_list, is_connected_list = check_validity(init_nodes, init_edges, is_valid_list, is_connected_list)

                if params.property_type is not None:
                    avg_property_loss = float(property_loss)/float(num_data_points)
                if params.loss_normalisation_type == 'by_total':
                    if params.embed_hs is True:
                        num_components += (total_nodes * 5)
                    num_components = float(total_nodes) + float(total_edges)
                    avg_val_loss = float(val_loss)/float(num_components)
                    if params.target_data_structs in ['nodes', 'both', 'random'] and total_nodes > 0:
                        avg_node_loss = float(node_loss)/float(num_components)
                        if params.embed_hs is True:
                            avg_hydrogen_loss = float(hydrogen_loss) / num_components
                            avg_charge_loss = float(charge_loss) / num_components
                            avg_is_in_ring_loss = float(is_in_ring_loss) / num_components
                            avg_is_aromatic_loss = float(is_aromatic_loss) / num_components
                            avg_chirality_loss = float(chirality_loss) / num_components
                    if params.target_data_structs in ['edges', 'both', 'random'] and total_edges > 0:
                        avg_edge_loss = float(edge_loss)/float(num_components)
                elif params.loss_normalisation_type == 'by_component':
                    avg_val_loss = 0
                    if params.target_data_structs in ['nodes', 'both', 'random'] and total_nodes > 0:
                        avg_node_loss = float(node_loss)/float(total_nodes)
                        avg_val_loss += avg_node_loss
                        if params.embed_hs is True:
                            avg_hydrogen_loss = float(hydrogen_loss) / float(total_hydrogens)
                            avg_charge_loss = float(charge_loss) / float(total_nodes)
                            avg_is_in_ring_loss = float(is_in_ring_loss) / float(total_nodes)
                            avg_is_aromatic_loss = float(is_aromatic_loss) / float(total_nodes)
                            avg_chirality_loss = float(chirality_loss) / float(total_nodes)
                            avg_val_loss += avg_hydrogen_loss + avg_charge_loss + avg_is_in_ring_loss + \
                                            avg_is_aromatic_loss + avg_chirality_loss
                    if params.target_data_structs in ['edges', 'both', 'random'] and total_edges > 0:
                        avg_edge_loss = float(edge_loss)/float(total_edges)
                        avg_val_loss += avg_edge_loss
                    if params.property_type is not None:
                        avg_val_loss += avg_property_loss
                logger.info('Average validation loss: {0:.2f}'.format(avg_val_loss))
                val_iter = total_iter // params.val_after

                if params.check_pred_validity is True:
                    percent_valid = np.mean(is_valid_list) * 100
                    valid_percent_connected = np.mean(is_connected_list) * 100
                    logger.info('Percent valid: {}%'.format(percent_valid))
                    logger.info('Percent of valid molecules connected: {}%'.format(valid_percent_connected))

                results_dict = {'loss': avg_val_loss}

                if params.target_data_structs in ['nodes', 'both', 'random'] and total_nodes > 0:
                    nodes_acc, noncarbon_nodes_acc, carbon_nodes_acc, mask_node_acc, replace_node_acc, recon_node_acc =\
                        accuracies_from_totals({nodes_correct: total_nodes, noncarbon_nodes_correct: total_noncarbon_nodes,
                        carbon_nodes_correct: total_carbon_nodes, mask_nodes_correct: total_mask_nodes,
                        replace_nodes_correct: total_replace_nodes, recon_nodes_correct: total_recon_nodes})
                    results_dict.update({'nodes_acc': nodes_acc, 'carbon_nodes_acc': carbon_nodes_acc,
                                         'noncarbon_nodes_acc': noncarbon_nodes_acc, 'mask_node_acc': mask_node_acc,
                                         'replace_node_acc': replace_node_acc, 'recon_node_acc': recon_node_acc,
                                         'node_loss': avg_node_loss})
                    logger.info('Node loss: {0:.2f}'.format(avg_node_loss))
                    logger.info('Node accuracy: {0:.2f}%'.format(nodes_acc))
                    logger.info('Non-Carbon Node accuracy: {0:.2f}%'.format(noncarbon_nodes_acc))
                    logger.info('Carbon Node accuracy: {0:.2f}%'.format(carbon_nodes_acc))
                    logger.info('mask_node_acc {:.2f}, replace_node_acc {:.2f}, recon_node_acc {:.2f}'.format(
                                mask_node_acc, replace_node_acc, recon_node_acc))
                    if params.embed_hs is True:
                        hydrogen_acc, mask_hydrogen_acc, replace_hydrogen_acc, recon_hydrogen_acc = accuracies_from_totals(
                        {hydrogens_correct: total_hydrogens, mask_hydrogens_correct: total_mask_hydrogens,
                        replace_hydrogens_correct: total_replace_hydrogens, recon_hydrogens_correct: total_recon_hydrogens})
                        results_dict.update({'hydrogen_acc': hydrogen_acc, 'mask_hydrogen_acc': mask_hydrogen_acc,
                                             'replace_hydrogen_acc': replace_hydrogen_acc,
                                             'recon_hydrogen_acc': recon_hydrogen_acc, 'hydrogen_loss': avg_hydrogen_loss})
                        logger.info('Hydrogen loss: {0:.2f}'.format(avg_hydrogen_loss))
                        logger.info('Hydrogen accuracy: {0:.2f}%'.format(hydrogen_acc))
                        logger.info('mask_hydrogen_acc {:.2f}, replace_hydrogen_acc {:.2f}, recon_hydrogen_acc {:.2f}'.format(
                            mask_hydrogen_acc, replace_hydrogen_acc, recon_hydrogen_acc))

                        results_dict.update({'charge_loss': avg_charge_loss, 'is_in_ring_loss': avg_is_in_ring_loss,
                                             'is_aromatic_loss': avg_is_aromatic_loss,
                                             'chirality_loss': avg_chirality_loss})
                        logger.info('Charge loss: {0:.2f}'.format(avg_charge_loss))
                        logger.info('Is in ring loss: {0:.2f}'.format(avg_is_in_ring_loss))
                        logger.info('Is aromatic loss: {0:.2f}'.format(avg_is_aromatic_loss))
                        logger.info('Chirality loss: {0:.2f}'.format(avg_chirality_loss))

                if params.target_data_structs in ['edges', 'both', 'random'] and total_edges > 0:
                    edges_acc, no_edge_acc, edge_present_acc, mask_edge_acc, replace_edge_acc, recon_edge_acc =\
                        accuracies_from_totals({edges_correct: total_edges, no_edge_correct: total_no_edges,
                        edge_present_correct: total_edges_present, mask_edges_correct: total_mask_edges,
                        replace_edges_correct: total_replace_edges, recon_edges_correct: total_recon_edges})
                    results_dict.update({'edges_acc': edges_acc, 'edge_present_acc': edge_present_acc,
                        'no_edge_acc': no_edge_acc, 'mask_edge_acc': mask_edge_acc, 'replace_edge_acc': replace_edge_acc,
                        'recon_edge_acc': recon_edge_acc, 'edge_loss': avg_edge_loss})
                    logger.info('Edge loss: {0:.2f}'.format(avg_edge_loss))
                    logger.info('Edge accuracy: {0:.2f}%'.format(edges_acc))
                    logger.info('Edge present accuracy: {0:.2f}%'.format(edge_present_acc))
                    logger.info('No edge accuracy: {0:.2f}%'.format(no_edge_acc))
                    logger.info(" mask_edge_acc {:.2f}, replace_edge_acc {:.2f}, recon_edge_acc {:.2f}".format(
                                mask_edge_acc, replace_edge_acc, recon_edge_acc))
                    logger.info('\n')
                    for i in range(distributions.shape[0]):
                        for j in range(distributions.shape[1]):
                            logger.info(dist_names[i][j] + ':\t' + str(distributions[i, j, :]))
                        logger.info('\n')

                if params.property_type is not None:
                    results_dict.update({'property_loss': avg_property_loss})
                    logger.info('Property loss: {0:.2f}%'.format(avg_property_loss))

                if params.run_perturbation_analysis is True:
                    preds = run_perturbations(perturbation_loader, model, params.embed_hs,
                                              params.max_hs, params.perturbation_batch_size, params.local_cpu)
                    stats, percentages = aggregate_stats(preds)
                    logger.info('Percentages: {}'.format( pp.pformat(percentages) ))

                if params.tensorboard:
                    if params.run_perturbation_analysis is True:
                        writer.add_scalars('dev/perturbation_stability', {str(key): val for key, val in
                                                                          percentages.items()}, val_iter)
                    write_tensorboard(writer, 'dev', results_dict, val_iter)

                if params.gen_num_samples > 0:
                    calculate_gen_benchmarks(generator, params.gen_num_samples, training_smiles, logger)

                logger.info("----------------------------------")

                model_state_dict = model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict()

                best_loss = save_checkpoints(total_iter, avg_val_loss, best_loss, model_state_dict, opt.state_dict(),
                                             exp_path, logger, params.no_save, params.save_all)

                # Reset random seed
                set_seed_if(params.seed)
                logger.info('Validation complete')
            total_iter += 1

if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # run experiment
    main(params)
