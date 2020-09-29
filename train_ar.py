import argparse
import os
import random
import sys

import numpy as np
import torch
from guacamol.distribution_learning_benchmark import ValidityBenchmark, UniquenessBenchmark, KLDivBenchmark
from torch.nn.utils import clip_grad_norm_

# guacamol stuff
from ar_mock_generator import ARMockGenerator, generate_smiles
from src.data.loader import load_smiles_data
from src.logger import create_logger
from src.model.transformer import TransformerModel
from src.utils import bool_flag, set_seed, get_optimizer


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Language transfer")

    # main parameters
    parser.add_argument("--dump_path", type=str, default='dumped/', help="Experiment dump path")
    parser.add_argument("--multi_gpu", type=bool_flag, default=False,
                        help="Train model on multiple gpus")
    parser.add_argument("--exp_name", type=str, default="",
                        help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="",
                        help="Experiment ID")
    parser.add_argument("--data_path", type=str, default="",
                        help="Data path for QM9, train data path for ChEMBL")
    parser.add_argument("--val_data_path", type=str, default="",
                        help="Validation data path for ChEMBL")
    parser.add_argument("--load_path", type=str, default=None,
                        help="Path for loading checkpoint")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed")
    parser.add_argument("--graph_type", default=None)
    parser.add_argument("--data_type", choices=['QM9', 'ChEMBL'], default='ChEMBL')
    parser.add_argument("--num_node_types", type=int, default=None)
    parser.add_argument("--num_edge_types", type=int, default=None)
    parser.add_argument("--max_nodes", type=int, default=None)
    parser.add_argument("--use_smiles", action='store_true', help='Use smiles representation of the molecules')
    parser.add_argument("--targets", action='store_true')
    parser.add_argument("--max_vocab", type=int, default=-1,
                        help="Maximum vocabulary size (-1 to disable)")
    parser.add_argument("--min_count", type=int, default=0,
                        help="Minimum vocabulary count")
    # model parameters
    parser.add_argument("--emb_dim", type=int, default=512,
                        help="Embedding layer size")
    parser.add_argument("--n_layers", type=int, default=4,
                        help="Number of Transformer layers")
    parser.add_argument("--n_heads", type=int, default=8,
                        help="Number of Transformer heads")
    parser.add_argument("--dropout", type=float, default=0,
                        help="Dropout")
    parser.add_argument("--attention_dropout", type=float, default=0,
                        help="Dropout in the attention layer")
    parser.add_argument("--gelu_activation", type=bool_flag, default=False,
                        help="Use a GELU activation instead of ReLU")
    parser.add_argument("--share_inout_emb", type=bool_flag, default=True,
                        help="Share input and output embeddings")
    parser.add_argument("--sinusoidal_embeddings", type=bool_flag, default=False,
                        help="Use sinusoidal embeddings")

    # adaptive softmax
    parser.add_argument("--asm", type=bool_flag, default=False,
                        help="Use adaptive softmax")
    if parser.parse_known_args()[0].asm:
        parser.add_argument("--asm_cutoffs", type=str, default="8000,20000",
                            help="Adaptive softmax cutoffs")
        parser.add_argument("--asm_div_value", type=float, default=4,
                            help="Adaptive softmax cluster sizes ratio")

    # optimizer parameters
    parser.add_argument("--split_data", type=bool_flag, default=False,
                        help="Split data across workers of a same node")
    parser.add_argument("--optimizer", type=str, default="adam,beta1=0.9,beta2=0.98,lr=0.0003",
                        help="Optimizer (SGD / RMSprop / Adam, etc.)")
    parser.add_argument("--clip_grad_norm", type=float, default=5.0)


    # batch parameters
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Number of sentences per batch")
    parser.add_argument("--bptt", type=int, default=1024,
                        help="Sequence length")
    parser.add_argument("--max_len", type=int, default=300,
                        help="Maximum length of sentences (after splitting into chars)")
    parser.add_argument("--tokens_per_batch", type=int, default=-1,
                        help="Number of tokens per batch")
    parser.add_argument("--max_batch_size", type=int, default=0,
                        help="Maximum number of sentences per batch (used in combination with tokens_per_batch, 0 to disable)")

    # beam search (for MT only)
    parser.add_argument("--beam_size", type=int, default=1,
                        help="Beam size, default = 1 (greedy decoding)")
    parser.add_argument("--length_penalty", type=float, default=1,
                        help="Length penalty, values < 1.0 favor shorter sentences, while values > 1.0 favor longer ones.")
    parser.add_argument("--early_stopping", type=bool_flag, default=False,
                        help="Early stopping, stop as soon as we have `beam_size` hypotheses, although longer ones may have better scores.")

    # evaluation
    parser.add_argument("--eval_only", type=bool_flag, default=False,
                        help="Only run evaluations")
    parser.add_argument("--epoch_size", type=int, default=100000,
                        help="Epoch size / evaluation frequency (-1 for parallel data size)")
    parser.add_argument("--print_after", type=int, default=5,
                        help="Print after certain number of iters")

    # refinement steps (for MT encoder model only)
    parser.add_argument("--refinement_steps", type=int, default=1,
                        help="Number of refinement steps for L->R decoding of encoder_only (TLM) model")

    # debug
    parser.add_argument('--local_cpu', action='store_true')
    parser.add_argument("--debug_train", type=bool_flag, default=False,
                        help="Use valid sets for train sets (faster loading)")
    parser.add_argument("--debug_slurm", type=bool_flag, default=False,
                        help="Debug multi-GPU / multi-node within a SLURM job")
    # debug params
    parser.add_argument('--debug_small', action='store_true', help='Debug on a very small version of full dataset')
    parser.add_argument('--debug_fixed', action='store_true', help='Debug by masking out fixed nodes and edges corresponding to index 0')
    parser.add_argument('--tensorboard', action='store_true', help='Use tensorboard')
    parser.add_argument('--log_train_steps', default=200, help='train steps to log')
    parser.add_argument('--val_after', default=1000, type=int, help='validate after how many steps')
    parser.add_argument('--max_steps', default=10e6, type=int, help='total number of steps')

    return parser

if 'post' not in torch.__version__:
    version = [int(i) for i in torch.__version__.split('.')]
else:
    version = [int(i) for i in torch.__version__.split('.')[:-1]]
if version[0] >= 1 and version[1] >= 2:
    index_method = 'bool'
else:
    index_method = 'byte'

def calculate_loss(model, batch, params, get_scores=False):
    # process data
    x, len = batch
    if params.local_cpu is False:
        x = x.cuda()
        len = len.cuda()

    # feedforward through model
    hid = model('fwd', x=x, lengths=len, causal=True)

    # target indixes to predict (depending on length)
    alen = torch.arange(len.max(), dtype=torch.long, device=len.device)
    pred_mask = alen[:, None] < len[None] - 1  # do not predict anything given the last target word
    y = x[1:].masked_select(pred_mask[:-1])
    assert y.shape[0] == (len - 1).sum().item()

    # loss
    if get_scores:
        scores, loss = model('predict', tensor=hid, pred_mask=pred_mask, y=y, get_scores=True)
        return scores, loss, y
    else:
        _, loss = model('predict', tensor=hid, pred_mask=pred_mask, y=y, get_scores=False)
        return loss

def save_model(params, data, model, opt, dico, logger, name, epoch, total_iter, scores):
    """
    Save the model.
    """
    path = os.path.join(params.dump_path, params.exp_name, '%s.pth' % name)
    logger.info('Saving models to %s ...' % path)
    data = {}
    # save actual model
    if params.multi_gpu:
        data['model'] = model.module.state_dict()
    else:
        data['model'] = model.state_dict()
    data['optimizer'] = opt.state_dict()
    # save optimizer parameters as well
    data['dico_id2word'] = dico.id2word
    data['dico_word2id'] = dico.word2id
    data['dico_counts'] = dico.counts
    data['params'] = {k: v for k, v in params.__dict__.items()}
    data['epoch'] = epoch
    data['total_iter'] = total_iter
    data['scores'] = scores

    torch.save(data, path)

def load_model(params, model, opt, logger):
    logger.info('Reloading models from %s ...' % params.load_path)
    data = torch.load(params.load_path, map_location=lambda storage, loc: storage.cuda(-1))

    model.load_state_dict(data['model'])
    opt.load_state_dict(data['optimizer'])
    logger.info('Done reloading model')
    return data['total_iter'], data['scores']

def main(params):
    # setup random seeds
    set_seed(params.seed)
    params.ar = True

    exp_path = os.path.join(params.dump_path, params.exp_name)
    # create exp path if it doesn't exist
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    # create logger
    logger = create_logger(os.path.join(exp_path, 'train.log'), 0)
    logger.info("============ Initialized logger ============")
    logger.info("Random seed is {}".format(params.seed))
    logger.info("\n".join("%s: %s" % (k, str(v))
                          for k, v in sorted(dict(vars(params)).items())))
    logger.info("The experiment will be stored in %s\n" % exp_path)
    logger.info("Running command: %s" % 'python ' + ' '.join(sys.argv))
    logger.info("")
    # load data
    data, loader = load_smiles_data(params)
    if params.data_type == 'ChEMBL':
        all_smiles_mols = open(os.path.join(params.data_path, 'guacamol_v1_all.smiles'), 'r').readlines()
    else:
        all_smiles_mols = open(os.path.join(params.data_path, 'QM9_all.smiles'), 'r').readlines()
    train_data, val_data = data['train'], data['valid']
    dico = data['dico']
    logger.info ('train_data len is {}'.format(len(train_data)))
    logger.info ('val_data len is {}'.format(len(val_data)))

    # keep cycling through train_loader forever
    # stop when max iters is reached
    def rcycle(iterable):
        saved = []                 # In-memory cache
        for element in iterable:
            yield element
            saved.append(element)
        while saved:
            random.shuffle(saved)  # Shuffle every batch
            for element in saved:
                  yield element
    train_loader = rcycle(train_data.get_iterator(shuffle=True, group_by_size=True, n_sentences=-1))

    # extra param names for transformermodel
    params.n_langs = 1
    # build Transformer model
    model = TransformerModel(params, is_encoder=False, with_output=True)

    if params.local_cpu is False:
        model = model.cuda()
    opt = get_optimizer(model.parameters(), params.optimizer)
    scores = {'ppl': np.float('inf'), 'acc': 0}

    if params.load_path:
        reloaded_iter, scores = load_model(params, model, opt, logger)

    for total_iter, train_batch in enumerate(train_loader):
        if params.load_path is not None:
            total_iter += reloaded_iter + 1

        epoch = total_iter // params.epoch_size
        if total_iter == params.max_steps:
            logger.info("============ Done training ... ============")
            break
        elif total_iter % params.epoch_size == 0:
            logger.info("============ Starting epoch %i ... ============" % epoch)
        model.train()
        opt.zero_grad()
        train_loss = calculate_loss(model, train_batch, params)
        train_loss.backward()
        if params.clip_grad_norm > 0:
            clip_grad_norm_(model.parameters(), params.clip_grad_norm)
        opt.step()
        if total_iter % params.print_after == 0:
            logger.info("Step {} ; Loss = {}".format(total_iter, train_loss))

        if total_iter > 0 and total_iter % params.epoch_size == (params.epoch_size - 1):
            # run eval step (calculate validation loss)
            model.eval()
            n_chars = 0
            xe_loss = 0
            n_valid = 0
            logger.info("============ Evaluating ... ============")
            val_loader = val_data.get_iterator(shuffle=True)
            for val_iter, val_batch in enumerate(val_loader):
                with torch.no_grad():
                    val_scores, val_loss, val_y = calculate_loss(model, val_batch, params, get_scores=True)
                # update stats
                n_chars += val_y.size(0)
                xe_loss += val_loss.item() * len(val_y)
                n_valid += (val_scores.max(1)[1] == val_y).sum().item()

            ppl = np.exp(xe_loss / n_chars)
            acc = 100. * n_valid / n_chars
            logger.info("Acc={}, PPL={}".format(acc, ppl))
            if acc > scores['acc']:
                scores['acc'] = acc
                scores['ppl'] = ppl
                save_model(params, data, model, opt, dico, logger, 'best_model', epoch, total_iter, scores)
                logger.info('Saving new best_model {}'.format(epoch))
                logger.info("Best Acc={}, PPL={}".format(scores['acc'], scores['ppl']))

            logger.info("============ Generating ... ============")
            number_samples = 100
            gen_smiles = generate_smiles(params, model, dico, number_samples)
            generator = ARMockGenerator(gen_smiles)

            try:
                benchmark = ValidityBenchmark(number_samples=number_samples)
                validity_score = benchmark.assess_model(generator).score
            except:
                validity_score = -1
            try:
                benchmark = UniquenessBenchmark(number_samples=number_samples)
                uniqueness_score = benchmark.assess_model(generator).score
            except:
                uniqueness_score = -1

            try:
                benchmark = KLDivBenchmark(number_samples=number_samples, training_set=all_smiles_mols)
                kldiv_score = benchmark.assess_model(generator).score
            except:
                kldiv_score = -1
            logger.info('Validity Score={}, Uniqueness Score={}, KlDiv Score={}'.format(validity_score, uniqueness_score, kldiv_score))
            save_model(params, data, model, opt, dico, logger, 'model', epoch, total_iter, {'ppl': ppl, 'acc': acc})
if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # run experiment
    main(params)
