import os
import pprint
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from src.data.loader import load_graph_data
from src.logger import create_logger
from src.model.gnn import MODELS_DICT
from src.model.graph_generator import GraphGenerator
from src.utils import set_seed_if, get_optimizer, get_index_method, write_tensorboard, lr_decay_multiplier, \
    save_checkpoints, calculate_gen_benchmarks, get_loss, dct_to_cuda_inplace, normalise_loss
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

    logger.info ('train_loader len is {}'.format(len(train_loader)))
    logger.info ('val_loader len is {}'.format(len(val_loader)))

    if params.num_binary_graph_properties > 0 and params.pretrained_property_embeddings_path:
        model.binary_graph_property_embedding_layer.weight.data = \
            torch.Tensor(np.load(params.pretrained_property_embeddings_path).T)
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
            if hasattr(model, 'seq_model'): model.seq_model.eval()

            # Training step
            batch_init_graph, batch_orig_graph, batch_target_type_graph, _, \
            graph_properties, binary_graph_properties = train_batch
            # init is what goes into model
            # target_inds and target_coords are 1 at locations to be predicted, 0 elsewhere
            # target_inds and target_coords are now calculated here rather than in dataloader
            # original are uncorrupted data (used for comparison with prediction in loss calculation)
            # masks are 1 in places corresponding to nodes that exist, 0 in other places (which are empty/padded)
            # target_types are 1 in places to mask, 2 in places to replace, 3 in places to reconstruct, 0 in places not to predict

            if params.local_cpu is False:
                batch_init_graph = batch_init_graph.to(torch.device('cuda:0'))
                batch_orig_graph = batch_orig_graph.to(torch.device('cuda:0'))
                dct_to_cuda_inplace(graph_properties)
                if binary_graph_properties: binary_graph_properties = binary_graph_properties.cuda()

            if grad_accum_iters % params.grad_accum_iters == 0:
                opt.zero_grad()

            _, batch_scores_graph, graph_property_scores = model(batch_init_graph, graph_properties,
                binary_graph_properties)

            num_components = sum([(v != 0).sum().numpy() for _, v in batch_target_type_graph.ndata.items()]) + \
                             sum([(v != 0).sum().numpy() for _, v in batch_target_type_graph.edata.items()])

            # calculate score
            losses = []
            results_dict = {}
            metrics_to_print = []
            if params.target_data_structs in ['nodes', 'both', 'random'] and \
                    batch_target_type_graph.ndata['node_type'].sum() > 0:
                node_losses = {}
                for name, target_type in batch_target_type_graph.ndata.items():
                    node_losses[name] = get_loss(
                        batch_orig_graph.ndata[name][target_type.numpy() != 0],
                        batch_scores_graph.ndata[name][target_type.numpy() != 0],
                        params.equalise, params.loss_normalisation_type, num_components, params.local_cpu)
                losses.extend(node_losses.values())
                results_dict.update(node_losses)
                metrics_to_print.extend(node_losses.keys())

            if params.target_data_structs in ['edges', 'both', 'random'] and \
                    batch_target_type_graph.edata['edge_type'].sum() > 0:
                edge_losses = {}
                for name, target_type in batch_target_type_graph.edata.items():
                    edge_losses[name] = get_loss(
                        batch_orig_graph.edata[name][target_type.numpy() != 0],
                        batch_scores_graph.edata[name][target_type.numpy() != 0],
                        params.equalise, params.loss_normalisation_type, num_components, params.local_cpu)
                losses.extend(edge_losses.values())
                results_dict.update(edge_losses)
                metrics_to_print.extend(edge_losses.keys())

            if params.predict_graph_properties is True:
                graph_property_losses = {}
                for name, scores in graph_property_scores.items():
                    graph_property_loss = normalise_loss(F.mse_loss(scores, graph_properties[name], reduction='sum'),
                                                         len(scores), num_components, params.loss_normalisation_type)
                    graph_property_losses[name] = graph_property_loss
                losses.extend(graph_property_losses.values())
                results_dict.update(graph_property_losses)
                metrics_to_print.extend(graph_property_losses.keys())

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


            if total_iter > 0 and total_iter % params.val_after == 0:
                logger.info('Validating')
                val_loss, num_data_points = 0, 0
                node_property_losses = {name: 0 for name in train_data.node_property_names}
                edge_property_losses = {name: 0 for name in train_data.edge_property_names}
                node_property_num_components = {name: 0 for name in train_data.node_property_names}
                edge_property_num_components = {name: 0 for name in train_data.edge_property_names}
                graph_property_losses = {name: 0 for name in params.graph_property_names}
                model.eval()
                set_seed_if(params.seed)
                for batch_init_graph, batch_orig_graph, batch_target_type_graph, _, \
                    graph_properties, binary_graph_properties in val_loader:

                    if params.local_cpu is False:
                        batch_init_graph = batch_init_graph.to(torch.device('cuda:0'))
                        batch_orig_graph = batch_orig_graph.to(torch.device('cuda:0'))
                        dct_to_cuda_inplace(graph_properties)
                        if binary_graph_properties: binary_graph_properties = binary_graph_properties.cuda()

                    with torch.no_grad():
                        _, batch_scores_graph, graph_property_scores = model(batch_init_graph, graph_properties,
                                                                             binary_graph_properties)

                    num_data_points += float(batch_orig_graph.batch_size)
                    losses = []
                    if params.target_data_structs in ['nodes', 'both', 'random'] and \
                            batch_target_type_graph.ndata['node_type'].sum() > 0:
                        for name, target_type in batch_target_type_graph.ndata.items():
                            iter_node_property_loss = F.cross_entropy(
                                batch_scores_graph.ndata[name][target_type.numpy() != 0],
                                batch_orig_graph.ndata[name][target_type.numpy() != 0], reduction='sum').cpu().item()
                            node_property_losses[name] += iter_node_property_loss
                            losses.append(iter_node_property_loss)
                            node_property_num_components[name] += float((target_type != 0).sum())

                    if params.target_data_structs in ['edges', 'both', 'random'] and \
                            batch_target_type_graph.edata['edge_type'].sum() > 0:
                        for name, target_type in batch_target_type_graph.edata.items():
                            iter_edge_property_loss = F.cross_entropy(
                                batch_scores_graph.edata[name][target_type.numpy() != 0],
                                batch_orig_graph.edata[name][target_type.numpy() != 0], reduction='sum').cpu().item()
                            edge_property_losses[name] += iter_edge_property_loss
                            losses.append(iter_edge_property_loss)
                            edge_property_num_components[name] += float((target_type != 0).sum())

                    if params.predict_graph_properties is True:
                        for name, scores in graph_property_scores.items():
                            iter_graph_property_loss = F.mse_loss(
                                scores, graph_properties[name], reduction='sum').cpu().item()
                            graph_property_losses[name] += iter_graph_property_loss
                            losses.append(iter_graph_property_loss)

                    val_loss += sum(losses)

                avg_node_property_losses, avg_edge_property_losses, avg_graph_property_losses = {}, {}, {}
                if params.loss_normalisation_type == 'by_total':
                    total_num_components = float(sum(node_property_num_components.values()) +
                                                 sum(edge_property_num_components.values()))
                    avg_val_loss = val_loss/total_num_components
                    if params.target_data_structs in ['nodes', 'both', 'random']:
                        for name, loss in node_property_losses.items():
                            avg_node_property_losses[name] = loss/total_num_components
                    if params.target_data_structs in ['edges', 'both', 'random']:
                        for name, loss in edge_property_losses.items():
                            avg_edge_property_losses[name] = loss/total_num_components
                    if params.predict_graph_properties is True:
                        for name, loss in graph_property_losses.items():
                            avg_graph_property_losses[name] = loss/total_num_components
                elif params.loss_normalisation_type == 'by_component':
                    avg_val_loss = 0
                    if params.target_data_structs in ['nodes', 'both', 'random']:
                        for name, loss in node_property_losses.items():
                            avg_node_property_losses[name] = loss/node_property_num_components[name]
                        avg_val_loss += sum(avg_node_property_losses.values())
                    if params.target_data_structs in ['edges', 'both', 'random']:
                        for name, loss in edge_property_losses.items():
                            avg_edge_property_losses[name] = loss/edge_property_num_components[name]
                        avg_val_loss += sum(avg_edge_property_losses.values())
                    if params.predict_graph_properties is True:
                        for name, loss in graph_property_losses.items():
                            avg_graph_property_losses[name] = loss/num_data_points
                        avg_val_loss += sum(avg_graph_property_losses.values())
                val_iter = total_iter // params.val_after

                results_dict = {'Validation_loss': avg_val_loss}
                for name, loss in avg_node_property_losses.items():
                    results_dict['{}_loss'.format(name)] = loss
                for name, loss in avg_edge_property_losses.items():
                    results_dict['{}_loss'.format(name)] = loss
                for name, loss in avg_graph_property_losses.items():
                    results_dict['{}_loss'.format(name)] = loss

                for name, loss in results_dict.items():
                    logger.info('{}: {:.2f}'.format(name, loss))

                if params.tensorboard:
                    write_tensorboard(writer, 'Dev', results_dict, val_iter)
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
