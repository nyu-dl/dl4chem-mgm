[![DOI](https://zenodo.org/badge/299354589.svg)](https://zenodo.org/badge/latestdoi/299354589)

# Masked Graph Modeling for Molecule Generation

This repository and its references contain the models, data and scripts used to carry out the experiments in the
[Masked graph modeling for molecule generation](https://www.nature.com/articles/s41467-021-23415-2) paper.

## Installation Guide

We used a Linux OS with an Nvidia Tesla P100-SXM2 GPU with 16 GB of memory.

A conda environment file (environment.yml) is provided as part of this repository. It may contain packages beyond those
needed to run the scripts here. If not using this file, please install the following dependencies.

### Python

python 3.7 \
pytorch 1.4.0 \
tensorflow 1.14 \
rdkit 2019.09.3 \
guacamol 0.5.0 \
dgl 0.4.3post2 \
tensorboardx 2.0 \
scipy 1.4.1

### GPU
CUDA (We used version 10.0) \
Pytorch, Tensorflow and dgl installations should correspond to the CUDA version used.

## Datasets
### QM9
QM9 SMILES strings are included in this repository at data/QM9/QM9_smiles.txt \
To process QM9 smiles for use in train and generation scripts:\
`python -m data.gen_targets --data-path data/QM9/QM9_smiles.txt --save-path data/QM9/QM9_processed.p --dataset-type QM9`
### ChEMBL
To download the ChEMBL dataset:\
Training set: `wget -O data/ChEMBL/guacamol_v1_train.smiles https://ndownloader.figshare.com/files/13612760`\
Validation set: `wget -O data/ChEMBL/guacamol_v1_valid.smiles https://ndownloader.figshare.com/files/13612766`\
Full dataset (training + validation + test): `wget -O data/ChEMBL/guacamol_v1_all.smiles https://ndownloader.figshare.com/files/13612745`

To process the training dataset after downloading for use in train and generation scripts:\
`python -m data.gen_targets --data-path data/ChEMBL/guacamol_v1_train.smiles
--save-path data/ChEMBL/ChEMBL_train_processed.p --dataset-type ChEMBL`\
To process the validation dataset after downloading for use in train and generation scripts:\
`python -m data.gen_targets --data-path data/ChEMBL/guacamol_v1_valid.smiles
--save-path data/ChEMBL/ChEMBL_val_processed.p --dataset-type ChEMBL`

## Pretrained Models
Pretrained models are provided
[here](https://drive.google.com/drive/folders/1J-DvXcUGjyeDbs_08c08vAuVMTViEofP?usp=sharing) for both datasets.
To use these for generation, download the entire `dumped` folder to the repository root.

## Training
As an alternative to using pretrained models, the following are scripts for training models from scratch.

### QM9
`python train.py --data_path data/QM9/QM9_processed.p --graph_type QM9 --exp_name QM9_experiment
--num_node_types 5 --num_edge_types 5 --max_nodes 9 --layer_norm --spatial_msg_res_conn
--batch_size 1024 --val_batch_size 2500 --val_after 105 --num_epochs 200 --shuffle
--mask_independently --force_mask_predict --optimizer adam,lr=0.0001 --tensorboard`

### ChEMBL
`python train.py --data_path data/ChEMBL/ChEMBL_train_processed.p --graph_type ChEMBL --exp_name chembl_experiment
--val_data_path data/ChEMBL/ChEMBL_val_processed.p --num_node_types 12 --num_edge_types 5 --max_nodes 88
--min_charge -1 --max_charge 3 --mpnn_steps 6 --layer_norm --spatial_msg_res_conn --batch_size 32 --val_batch_size 64
--grad_accum_iters 16 --val_after 3200 --num_epochs 10 --shuffle --force_mask_predict --mask_independently
--optimizer adam,lr=0.0001 --tensorboard`

## Training Baseline Transformer Models
In addition to scripts for training our model, we include the scripts for training the baseline autoregressive
Transformer models (download preprocessed data for the Transformer models
[here](https://drive.google.com/drive/folders/16PDJlpI1HnTy-D7ZnUX-_VkEBqpcM093?usp=sharing)). The code for the model
is based on the following repository: https://github.com/facebookresearch/XLM

### ChEMBL (Transformer Regular)

`python train_ar.py --dump_path ./ --data_path /path/to/chembl/data --data_type ChEMBL --exp_name chembl_smiles_transformer_regular --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 --epoch_size 1000 --emb_dim 1024 --n_layers 6 --n_heads 8 --dropout 0.1 --attention_dropout 0.1 --batch_size 128 `

### ChEMBL (Transformer Small)

`python train_ar.py --dump_path ./ --data_path /path/to/chembl/data --data_type ChEMBL --exp_name chembl_smiles_transformer_small --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 --epoch_size 1000 --emb_dim 512 --n_layers 4 --n_heads 8 --dropout 0.1 --attention_dropout 0.1 --batch_size 128`

For QM9 take the above scripts ChEMBL and set the flag `--data_type` to `QM9`.

## Generation
The following are scripts for generating from trained/pretrained models.
The `--node_target_frac` and `--edge_target_frac` options set the masking rate for node and edge features respectively.

### QM9

To generate using training initialisation and masking rate 0.1:\
`python generate.py --data_path data/QM9/QM9_processed.p --graph_type QM9
--model_path dumped/QM9_experiment/best_model --smiles_dataset_path data/QM9/QM9_smiles.txt
--output_dir dumped/QM9_experiment/generation/train_init/mask10/results
--num_node_types 5 --num_edge_types 5 --max_nodes 9 --layer_norm --embed_hs --spatial_msg_res_conn
--num_iters 400 --num_sampling_iters 400
--cp_save_dir dumped/QM9_experiment/generation/train_init/mask10/generation_checkpoints --batch_size 2500
--checkpointing_period 400 --evaluation_period 20 --save_period 20 --evaluate_finegrained --save_finegrained
--mask_independently --retrieve_train_graphs --node_target_frac 0.1 --edge_target_frac 0.1`

To generate using marginal initialisation and masking rate 0.2:\
`python generate.py --data_path data/QM9/QM9_processed.p --graph_type QM9
--model_path dumped/QM9_experiment/best_model --smiles_dataset_path data/QM9/QM9_smiles.txt
--output_dir dumped/QM9_experiment/generation/marginal_init/mask20/results
--num_node_types 5 --num_edge_types 5 --max_nodes 9 --layer_norm --embed_hs --spatial_msg_res_conn
--num_iters 400 --num_sampling_iters 400
--cp_save_dir dumped/QM9_experiment/generation/train_init/mask20/generation_checkpoints --batch_size 2500
--checkpointing_period 400 --evaluation_period 20 --save_period 20 --evaluate_finegrained --save_finegrained
--mask_independently --random_init --node_target_frac 0.2 --edge_target_frac 0.2`

### ChEMBL

To generate using training initialisation and masking rate 0.01:\
`python generate.py --data_path data/ChEMBL/ChEMBL_train_processed.p --graph_type ChEMBL
--model_path dumped/chembl_experiment/best_model --smiles_dataset_path data/ChEMBL/guacamol_v1_all.smiles
--output_dir dumped/chembl_experiment/generation/train_init/mask1/results
--num_node_types 12 --num_edge_types 5 --max_nodes 88 --min_charge -1 --max_charge 3 --layer_norm --mpnn_steps 6
--embed_hs --spatial_msg_res_conn
--num_iters 300 --num_sampling_iters 300
--checkpointing_period 300 --evaluation_period 20 --save_period 20 --evaluate_finegrained --save_finegrained
--cp_save_dir dumped/chembl_experiment/generation/train_init/mask1/generation_checkpoints  --batch_size 32
--mask_independently --retrieve_train_graphs --node_target_frac 0.01 --edge_target_frac 0.01`

## Generation Using Transformer Baseline Models

### QM9

`python generate_ar_distributional.py --model_path /path/to/trained/qm9/model/best_model.pth \
  --dist_file QM9_all.smiles`

### ChEMBL

`python generate_ar_distributional.py --model_path /path/to/trained/chembl/model/best_model.pth \
  --dist_file guacamol_v1_all.smiles`


## MGM Generation Results
SMILES strings and distributional results at each recorded generation step can be found in <output_dir> from the
MGM generation script used above.

To print generation results at each step in a dataframe:
`python get_best_distributional_results.py <output_dir>`

We also provide a list of SMILES strings of 20,000 generated molecules each for QM9 with a 10% masking rate and ChEMBL 
with a 1% masking rate [here](https://drive.google.com/drive/folders/1SiOLr3RVr7wcgXUuGoPRn1I-iTwcn2k6?usp=sharing).
Training initialisation was used in both cases.

## Citation
If you have found the materials in this repository useful, please consider citing:
Mahmood, O., Mansimov, E., Bonneau, R. et al. Masked graph modeling for molecule generation. Nat Commun 12, 3156 (2021). https://doi.org/10.1038/s41467-021-23415-2