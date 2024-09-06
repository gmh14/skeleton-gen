# make deterministic
from mingpt.utils import set_seed

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import os

import math
from torch.utils.data import Dataset
from utils.data import SkeletonDatasetAutoRegr
import json
from mingpt.model import GPT, GPTUPD, GPTConfig
from mingpt.trainer import Trainer, TrainerConfig
import argparse
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def quantize_vals(vals, n_vals=256, shift=0.5):
    # print('quant vals', n_vals)
    delta = 1. / n_vals
    quant_vals = ((vals + shift) // delta).astype(np.int32)

    return quant_vals


def inv_quantize_vals(quant_vals, n_vals=256, shift=0.5):
    print('inv quant vals', n_vals)
    delta = 1. / n_vals
    vals = (quant_vals * delta - shift)

    return vals



set_seed(42)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


parser = argparse.ArgumentParser(description='Copy meshes data.')
parser.add_argument('--file_path', type=str,
                    help='Path to data', required=True)
parser.add_argument('--subsample', type=int, default=None)
parser.add_argument('--num_epoch', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--retrieval_key', type=str, default='tet_graph_points')
parser.add_argument('--ckpt_path', type=str, default='ckpt/')
parser.add_argument('--use_pretrained', action='store_true')
parser.add_argument('--use_cpu', action='store_true', default=False)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--train_combined', action='store_true')
parser.add_argument('--num_skeleton_points', type=int, default=256)
parser.add_argument('--num_surface_points', type=int, default=256)
parser.add_argument('--data_subsample', type=int, default=None)
parser.add_argument('--suffix', type=str, default='')
parser.add_argument('--cloud_order', type=str, default='mixed')
parser.add_argument('--shape_scale', type=float, default=1.04)
parser.add_argument('--num_quant_vals', type=int, default=1000)
parser.add_argument('--num_layers', type=int, default=8)
parser.add_argument('--num_heads', type=int, default=8)

args = parser.parse_args()
print(args)
gpus = torch.cuda.device_count()
print('GPUs available', gpus)

sort_device = 'cpu'

print('Sort device', sort_device)
ids_to_load = None #['39735'] #debugging mostly

train_dataset = SkeletonDatasetAutoRegr(args.file_path, subsample=args.subsample,
                                        num_skeleton_points=args.num_skeleton_points,
                                        data_subsample=args.data_subsample,
                                        select_random=True,
                                        cloud_order=args.cloud_order,
                                        shape_scale=args.shape_scale,
                                        num_tokens=args.num_quant_vals,
                                        sort_device=sort_device)

block_size = len(train_dataset[0][0]) + 1

print(train_dataset.skeletons.shape)
print('BLOCK SIZE', block_size)

print(train_dataset.num_tokens)
print(train_dataset[0])
#print(train_dataset.categories)
#print(train_dataset.categories_list)
mconf = GPTConfig(train_dataset.num_tokens, block_size,
                  n_layer=args.num_layers,
                  n_head=args.num_heads, n_embd=512)
model = GPTUPD(mconf)
model.register_buffer('pos_emb_inds', train_dataset.pos_emb_inds)
model.register_buffer('num_base_tokens', torch.LongTensor([train_dataset.num_base_tokens]))
model.register_buffer('num_tokens', torch.LongTensor([train_dataset.num_tokens]))
model.register_buffer('shape_scale', torch.FloatTensor([train_dataset.shape_scale]))
model.register_buffer('shape_scale', torch.FloatTensor([train_dataset.shape_scale]))
#model.pos_emb_inds = train_dataset.pos_emb_inds

# initialize a trainer instance and kick off training
model_suffix = args.file_path.split('/')[-1][:-4]
print(model_suffix)
model_suffix += args.suffix
model_name = f'{args.ckpt_path}/skeleton_cloud_generative_{args.cloud_order}_{model_suffix}.pt'
print(model_name)


tconf = TrainerConfig(max_epochs=args.num_epoch, batch_size=args.batch_size, learning_rate=6e-4,

                      lr_decay=True, warmup_tokens=512 * 20, final_tokens=2 * len(train_dataset) * block_size,
                      num_workers=0, use_pretrained=args.use_pretrained,
                      ckpt_path=model_name,
                      gpu_id=args.gpu_id, use_cpu=args.use_cpu)
trainer = Trainer(model, train_dataset, None, tconf)
trainer.train()
print('Done with script.')

