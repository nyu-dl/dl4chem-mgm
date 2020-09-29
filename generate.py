import os
import torch

from src.data.loader import load_graph_data
from src.model.gnn import MODELS_DICT
from src.model.graph_generator import GraphGenerator, MockGenerator, MockGDGenerator, \
    evaluate_uncond_generation
from train_script_parser import get_parser


def get_final_parser():
    parser = get_parser()

    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--smiles_dataset_path", type=str, default="", help='path to smiles strings for entire dataset')
    parser.add_argument("--gd_output_path", type=str, default="", help='directory path to save individual gd outputs')
    parser.add_argument('--json_output_path', default='', help='path to save overall results')
    parser.add_argument('--smiles_output_path', default='', help='path to save generated smiles strings')
    parser.add_argument('--output_dir', default='', help='directory to save results for checkpointed uncond generation')
    parser.add_argument('--cp_save_dir', default=None)
    parser.add_argument('--checkpointing_period', type=int, default=1, help='num iters between uncond checkpoints')
    parser.add_argument('--save_period', type=int, default=1, help='num iters between saving uncond generated smiles')
    parser.add_argument('--save_finegrained', action='store_true', help='save every iter for first 10 iters')
    parser.add_argument('--evaluation_period', type=int, default=1, help='num iters between uncond evaluations')
    parser.add_argument('--evaluate_finegrained', action='store_true', help='evaluate every iter for first 10 iters')
    parser.add_argument('--save_init', action='store_true', help='save smiles strings of initialised molecules '
                                                                 '(only for use with random_init)')
    parser.add_argument('--set_seed_at_load_iter', action='store_true')
    parser.add_argument("--num_samples_to_generate", type=int, default=20000)
    parser.add_argument("--num_samples_to_evaluate", type=int, default=10000)
    parser.add_argument('--generation_algorithm', choices=['gibbs', 'simultaneous'], default='simultaneous')
    parser.add_argument('--variables_per_gibbs_iteration', type=int, default=1, help='how many variables per graph to'
                                                                        'sample simultaneously if using gibbs sampling')
    parser.add_argument("--num_iters", type=int, default=3)
    parser.add_argument('--random_init', action='store_true')
    parser.add_argument('--sample_uniformly', action='store_true',
                        help='sample initial components from uniform instead of '
                             'categorical distribution, for use with random_init')
    parser.add_argument('--num_sampling_iters', type=int, default=0, help='number of iterations for which to use Gibbs'
                                                                          'Sampling before using argmax')
    parser.add_argument('--top_k', type=int, default=-1, help='value of k for top-k sampling')
    parser.add_argument('--maintain_minority_proportion', action='store_true',
                        help='force nodes initialised as non-carbon'
                             'to remain non-carbon')
    parser.add_argument('--retrieve_train_graphs', action='store_true',
                        help='use corrupted training set graphs as initial graphs')
    parser.add_argument('--mask_comp_to_predict', action='store_true',
                        help='mask component that will be predicted in each Gibbs sampling iteration')
    parser.add_argument('--one_property_per_loop', action='store_true', help='only replace one node property after each'
                                                                             'forward pass during gibbs sampling')
    parser.add_argument('--evaluate_connected_only', action='store_true')
    parser.add_argument('--use_mock_generator', action='store_true')
    parser.add_argument('--smiles_load_path', default=None, help='Path to smiles strings that have already been'
                                                                 'generated, for use with mock generator')

    return parser


def setup_data_and_generator(params):
    params.shuffle = False
    weighted_loss = not not params.weighted_loss
    params.weighted_loss = True
    output_dir = os.path.dirname(params.json_output_path) if params.json_output_path else params.output_dir
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if params.use_mock_generator is True:
        if params.goal_directed is True:
            generator = MockGDGenerator(params.smiles_load_path)
        else:
            with open(params.smiles_load_path) as f:
                smiles_list = f.read().strip().split('\n')
            generator = MockGenerator(smiles_list, params.num_samples_to_generate)
        if params.evaluate_connected_only is True:
           generator.smiles_list = [s for s in generator.smiles_list if '.' not in s]
    else:
        if params.cp_save_dir is not None and not os.path.isdir(params.cp_save_dir):
            os.mkdir(params.cp_save_dir)
        train_data, _, _, _ = load_graph_data(params)
        params.weighted_loss = weighted_loss
        model_cls = MODELS_DICT[params.model_name]
        model = model_cls(params)
        if params.local_cpu is True:
            model.load_state_dict(torch.load(params.model_path, map_location='cpu'))
        else:
            model.load_state_dict(torch.load(params.model_path))
            model = model.cuda()
        model.eval()
        generator = GraphGenerator(train_data, model, params.generation_algorithm, params.random_init, params.num_iters,
                                   params.num_sampling_iters, params.batch_size, params.edges_per_batch,
                                   params.retrieve_train_graphs, params.local_cpu, params.cp_save_dir,
                                   params.set_seed_at_load_iter, params.graph_type, params.sample_uniformly,
                                   params.mask_comp_to_predict, params.maintain_minority_proportion,
                                   params.no_edge_present_type, params.mask_independently, params.one_property_per_loop,
                                   params.checkpointing_period, params.save_period, params.evaluation_period,
                                   params.evaluate_finegrained, params.save_finegrained,
                                   params.variables_per_gibbs_iteration, params.top_k,
                                   params.save_init)

    return generator


def main(params):
    generator = setup_data_and_generator(params)
    if params.use_mock_generator is True:
        evaluate_uncond_generation(generator, params.smiles_dataset_path, params.json_output_path,
                                            params.num_samples_to_evaluate, params.evaluate_connected_only)
    else:
        _ = generator.generate_with_evaluation(params.num_samples_to_generate, params.smiles_dataset_path,
                                           params.output_dir, params.num_samples_to_evaluate,
                                           params.evaluate_connected_only)

if __name__ == '__main__':
    parser = get_final_parser()
    params = parser.parse_args()
    main(params)
