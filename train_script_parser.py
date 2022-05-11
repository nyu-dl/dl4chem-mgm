import argparse


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Language transfer")

    # main parameters
    parser.add_argument("--dump_path", type=str, default='dumped/',
                        help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="",
                        help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="",
                        help="Experiment ID")
    parser.add_argument("--data_path", type=str, default="",
                        help="Data path for QM9, train data path for ChEMBL")
    parser.add_argument("--val_data_path", type=str, default="data/ChEMBL/ChEMBL_val_processed_hs.p",
                        help="Validation data path for ChEMBL")
    parser.add_argument("--graph_properties_path", type=str, default="",
                        help="Data path for QM9, train data path for ChEMBL")
    parser.add_argument("--val_graph_properties_path", type=str, default="data/ChEMBL/ChEMBL_val_graph_properties.p",
                        help="Validation data path for ChEMBL")
    parser.add_argument('--graph2binary_properties_path', type=str, default='data/proteins/pdb_golabels.p')
    parser.add_argument('--val_graph2binary_properties_path', type=str, default=None)
    parser.add_argument('--pretrained_property_embeddings_path', type=str,
                        default='data/proteins/preprocessed_go_embeddings.npy', help='path to pretrained embeddings '
                                'for binary graph properties such as GO terms')
    parser.add_argument("--seed", type=int, default=-1,
                        help="random seed")
    parser.add_argument('--val_seed', type=int, default=None)
    parser.add_argument('--save_all', action='store_true', help='save all models, as opposed to only best and latest '
                                                                'models')
    parser.add_argument('--no_save', action='store_true', help='do not save any checkpoints')
    parser.add_argument('--load_latest', action='store_true', help='load latest model instead of training from scratch')
    parser.add_argument('--load_best', action='store_true', help='load best model instead of training from scratch')
    parser.add_argument('--first_iter', type=int, default=0, help='first iteration number, nonzero when loading '
                                                                  'checkpoint')

    # model and data parameters
    parser.add_argument('--model_name', choices=['GraphNN', 'SeqGraphNN'], default='GraphNN')
    parser.add_argument("--dim_h", type=int, default=2048, help="Hidden dimension size")
    parser.add_argument("--dim_k", type=int, default= 1, help="Max rank of edge matrices")
    parser.add_argument("--seq_output_dim", type=int, default=768,
                        help="dimensionality of sequence model output in SeqGraphNN")
    parser.add_argument('--use_newest_edges', action='store_true', help='In MPNN, use most edges from layer l+1 instead'
                                                                        'of l to update nodes in layer l')
    parser.add_argument("--graph_type", choices=['QM9', 'ChEMBL', 'protein'], default='QM9')
    parser.add_argument("--ar", action='store_true', help='Use autoregressive model (transformer) instead of graph neural network')
    parser.add_argument("--use_smiles", action='store_true', help='Use smiles representation of the molecules')
    parser.add_argument("--num_node_types", type=int, default=None)
    parser.add_argument("--num_edge_types", type=int, default=None, help='includes no edge as a type of edge')
    parser.add_argument('--no_edge_present_type', choices=['learned', 'zeros'], default='zeros', help='whether'
                                        'representation of no edge between two atoms is learned or a vector of zeros')
    parser.add_argument("--max_nodes", type=int, default=None)
    parser.add_argument('--share_embed', action='store_true', help='share embeddings and linear out weight')
    parser.add_argument("--embed_hs", action='store_true')
    parser.add_argument('--max_hs', type=int, default=4)
    parser.add_argument('--max_charge', type=int, default=1, help='must be nonnegative')
    parser.add_argument('--min_charge', type=int, default=-1, help='must be nonpositive')
    parser.add_argument('--mask_all_ring_properties', action='store_true', help='mask out is_aromatic and is_in_ring'
                                                                                'for all nodes in graph')
    parser.add_argument('--mask_independently', action='store_true', help='mask node properties independently of '
                                                                          'masked nodes')
    parser.add_argument("--property_type", choices=[None], default=None)
    parser.add_argument('--binary_classification', action='store_true', help='force node/edge data to contain only two '
                                                                             'categories')
    parser.add_argument("--equalise", action='store_true')
    parser.add_argument('--weighted_loss', action='store_true')
    parser.add_argument('--loss_normalisation_type', choices=['by_total', 'by_component'], default='by_component',
                        help='whether to normalise the total loss or normalise each loss component separately')
    parser.add_argument('--target_data_structs', choices=['both', 'nodes', 'edges', 'random'], default='both',
                        help='use nodes, edges or both as targets, or pick either randomly for each datapoint')
    parser.add_argument('--prediction_data_structs', choices=['all', 'random', 'nodes', 'edges'], default='all',
                        help='mark all available target data struct types for prediction, choose one at random, '
                             'or choose a specific type')
    parser.add_argument('--cond_virtual_node', action='store_true')
    parser.add_argument('--num_graph_properties', type=int, default=0)
    parser.add_argument('--graph_property_names', type=str, nargs='+', default=[])
    parser.add_argument('--normalise_graph_properties', action='store_true')
    parser.add_argument('--predict_graph_properties', action='store_true')
    parser.add_argument('--num_binary_graph_properties', type=int, default=0)

    # MPNN parameters
    parser.add_argument('--mpnn_name', choices=['EdgesOwnRepsMPNN', 'EdgesFromNodesMPNN'],
                        default='EdgesFromNodesMPNN', help='name of mpnn to use')
    parser.add_argument('--fully_connected', action='store_true', help='use fully connected graph in mpnns')
    parser.add_argument('--node_mpnn_name', choices=['MultiplicationMPNN', 'AdditionMPNN', 'TestMPNN',
                                    'NbrMultMPNN', 'NbrEWMultMPNN', 'DummyNodeMPNN'],
                                    default='NbrEWMultMPNN', help='name of node mpnn to use in mpnn')
    parser.add_argument('--update_edges_at_end_only', action='store_true', help='update edge representations in the'
                                                                                'final mpnn step only')
    parser.add_argument('--bound_edges', action='store_true', help='use sigmoid at end of edge_transform_network')
    parser.add_argument('--spatial_msg_res_conn', action='store_true', help='use spatial residual connection for nodes'
                                                                            'during message passing stage')
    parser.add_argument('--spatial_postgru_res_conn', action='store_true', help='use spatial residual connection for'
                                                                                 'nodes after GRU')
    parser.add_argument('--global_connection', action='store_true', help='include global connection from all nodes in'
                                                                            'graph during message passing stage')
    parser.add_argument("--num_mpnns", type=int, default=1)
    parser.add_argument("--mpnn_steps", type=int, default=4)
    parser.add_argument("--layer_norm", action='store_true')
    parser.add_argument("--res_conn", action='store_true')

    # Component corruption/prediction parameters
    parser.add_argument("--do_not_corrupt", action='store_true', help='do not corrupt data or set target components '
                                                                      'when retrieving from dataset')

    parser.add_argument("--node_target_frac", type=float, default=0.2)
    parser.add_argument("--edge_target_frac", type=float, default=0.2)
    parser.add_argument('--target_frac_type', choices=['fixed', 'random'], default='random',
                        help='use fixed target_frac or sample uniformly at random from [0, target_frac]')
    parser.add_argument('--target_frac_inc_amount', type=float, default=0, help='how much to increase target_frac for '
                                                                                'gradually increasing corruption level')
    parser.add_argument('--target_frac_inc_after', type=int, default=None, help='after how many iters to increase '
                                                                                'target_frac')
    parser.add_argument('--max_target_frac', type=float, default=0.8, help='stop increasing target_frac once it has'
                                                                           'reached this value')
    parser.add_argument("--val_node_target_frac", type=float, default=0.1)
    parser.add_argument("--val_edge_target_frac", type=float, default=0.1)
    parser.add_argument("--node_mask_frac", type=float, default=1.0, help='fraction of target nodes to mask')
    parser.add_argument("--edge_mask_frac", type=float, default=1.0, help='fraction of target nodes to mask')
    parser.add_argument("--node_replace_frac", type=float, default=0.0, help='fraction of target nodes to replace')
    parser.add_argument("--edge_replace_frac", type=float, default=0.0, help='fraction of target nodes to replace')
    parser.add_argument("--node_mask_predict_frac", type=float, default=1.0, help='fraction of masked nodes to predict')
    parser.add_argument("--edge_mask_predict_frac", type=float, default=1.0, help='fraction of masked nodes to predict')
    parser.add_argument("--node_replace_predict_frac", type=float, default=1.0, help='fraction of replaced nodes to '
                                                                                     'predict')
    parser.add_argument("--edge_replace_predict_frac", type=float, default=1.0, help='fraction of replaced nodes to '
                                                                                     'predict')

    parser.add_argument("--force_mask_predict", action='store_true', help='predict at least one masked component per '
                                                                          'graph')
    parser.add_argument("--force_replace_predict", action='store_true', help='predict at least one replaced component '
                                                                             'per graph')

    # MAT parameters
    parser.add_argument('--mat_N', type=int, default=2, help='number of dense layers in positionwise feedforward')
    parser.add_argument('--mat_d_model', type=int, default=64, help='model dimensionality')
    parser.add_argument('--mat_h', type=int, default=8, help='number of attention heads')
    parser.add_argument('--mat_dropout', type=float, default=0.1, help='dropout probability')

    # optimizer parameters
    parser.add_argument("--optimizer", type=str, default="adam,beta1=0.9,beta2=0.98,lr=0.0001",
                        help="Optimizer (SGD / RMSprop / Adam, etc.)")
    parser.add_argument("--clip_grad_norm", type=float, default=10.0)
    parser.add_argument("--no_update", action='store_true', help="don't update model parameters")
    parser.add_argument('--warm_up_iters', type=int, default=1.0, help='number of iterations over which lr warms up')
    parser.add_argument('--lr_decay_interval', type=int, default=9999999, help='number of iters after which lr decays')
    parser.add_argument('--lr_decay_frac', type=float, default=1.0, help='fraction of current lr that new lr is set'
                                                                           'to after decay')
    parser.add_argument('--lr_decay_amount', type=float, default=0.0, help='amount that lr decreases by at decay')
    parser.add_argument('--min_lr', type=float, default=0.0, help='minimum value of lr (as a result of decay)')
    parser.add_argument('--decay_start_iter', type=int, default=99999999, help='iteration at which to start lr decay')
    parser.add_argument('--grad_accum_iters', type=int, default=1, help='number of training iterations over which to'
                                                                        'accumulate gradient')


    # batch parameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument('--edges_per_batch', type=int, default=-1, help='use for batching by number of edges')
    parser.add_argument("--val_batch_size", type=int, default=None)
    parser.add_argument('--val_edges_per_batch', type=int, default=None, help='use for batching by number of edges')
    parser.add_argument("--perturbation_batch_size", type=int, default=32, help='batch size for perturbation loader')
    parser.add_argument('--perturbation_edges_per_batch', type=int, default=-1)
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of epochs")
    parser.add_argument('--shuffle', action='store_true', help='shuffle batches')

    # training parameters
    parser.add_argument("--max_epoch", type=int, default=100000,
                        help="Maximum epoch size")

    # debugging and logging parameters
    parser.add_argument('--local_cpu', action='store_true')
    parser.add_argument('--debug_small', action='store_true', help='Debug on a very small version of full dataset')
    parser.add_argument('--validate_on_train', action='store_true', help='if using debug_small, use train dataset as '
                                                                         'validation dataset')
    parser.add_argument('--num_batches', type=int, default=4, help='number of batches to use as dataset when using '
                                                                   'debug_small')
    parser.add_argument('--debug_fixed', action='store_true', help='Debug by masking out fixed nodes and edges '
                                                                   'corresponding to index 0')
    parser.add_argument('--tensorboard', action='store_true', help='Use tensorboard')
    parser.add_argument('--suppress_train_log', action='store_true', help='Do not log train results')
    parser.add_argument('--suppress_params', action='store_true', help='Do not log argparse params')
    parser.add_argument('--check_pred_validity', action='store_true', help='check if predicted molecules are valid')
    parser.add_argument('--log_train_steps', default=200, help='train steps to log')
    parser.add_argument('--val_after', default=1000, type=int, help='validate after how many steps')
    parser.add_argument('--val_dataset_size', default=-1, type=int, help='number of validation datapoints, use all '
                                                                         'if -1')
    parser.add_argument('--max_steps', default=10e6, type=int, help='validate after how many steps')

    # generation parameters
    parser.add_argument('--gen_num_samples', type=int, default=0)
    parser.add_argument('--gen_random_init', action='store_true')
    parser.add_argument('--gen_num_iters', type=int, default=10)
    parser.add_argument('--gen_predict_deterministically', action='store_true')
    parser.add_argument('--smiles_path', help='path to txt file of smiles strings of data')
    parser.add_argument('--smiles_train_split', type=float, default=0.8,
                        help='fraction of smiles strings in smiles_path file that correspond to training data')

    return parser
