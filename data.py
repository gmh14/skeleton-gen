# make deterministic
import numpy as np
import torch
import os
from collections import defaultdict

from torch.utils.data import Dataset
import json
# from .io import get_graph
import trimesh
from sklearn.neighbors import KDTree
import h5py
from torchvision.transforms import RandomResizedCrop


def quantize_vals(vals, n_vals=256, shift=0.5, shape_scale=1.0):
    # print('quant vals', n_vals)
    delta = shape_scale / n_vals
    quant_vals = ((vals + shift) // delta).astype(np.int32)

    return quant_vals


def inv_quantize_vals(quant_vals, n_vals=256, shift=0.5, shape_scale=1.0):
    print('inv quant vals', n_vals)
    delta = shape_scale / n_vals
    vals = (quant_vals * delta - shift)

    return vals


class SkeletonDatasetAutoRegr(Dataset):

    def __init__(self, file_path, num_points=511, num_tokens=128, subsample=None,
                 block_size=1536, data_subsample=None,
                 ids_to_load=None, load_sdf=False, num_skeleton_points=256,
                 cloud_order='mixed', select_random=True, shape_scale=1.04, sort_device=None,
                 sort_clouds=True):
        self.num_points = num_points
        self.block_size = block_size
        self.quant_clouds_skeleton = []
        self.quant_clouds_surface = []
        self.clouds_skeleton = []
        self.clouds_surface = []
        self.clouds = []
        self.tet_skeletons = []
        self.quant_clouds = []
        self.categories = []

        self.fin_ids = []

        # np.random.shuffle(self.ids)
        self.num_base_tokens = num_tokens
        self.load_sdf = load_sdf
        self.num_skeleton_points = num_skeleton_points

        data = np.load(file_path)

        self.categories_list = sorted(list(set(data['categories'])))
        self.categories_dict = dict(zip(self.categories_list, list(range(len(self.categories_list)))))
        print('NUM CATEGORIES')
        print(len(self.categories_list))
        print('Bounds check')
        print(data['skeletons'].max(), data['skeletons'].min())
        print('SHAPE SCALE')
        print(shape_scale)
        self.categories = data['categories'][:data_subsample]
        self.fin_ids = data['ids'][:data_subsample]
        self.sort_clouds = sort_clouds

        if sort_clouds:
            self.skeletons = quantize_vals(self.sort_clouds_in_4d(data['skeletons'][:data_subsample]),
                                           n_vals=self.num_base_tokens,
                                           shape_scale=shape_scale, shift=shape_scale/2)
        else:
            self.skeletons = quantize_vals(data['skeletons'][:data_subsample],
                                           n_vals=self.num_base_tokens,
                                           shape_scale=shape_scale, shift=shape_scale / 2)
        self.num_tokens = num_tokens + len(self.categories_list) + 3
        self.cloud_order = cloud_order
        self.select_random = select_random
        self.shape_scale = shape_scale
        print(f'Cloud order is {cloud_order}')
        cat_code_ind = torch.LongTensor([4])
        end_code_inds = torch.LongTensor([5])
        skeleton_pos_inds = torch.LongTensor(torch.arange(self.skeletons.shape[2] * 3) % 3)
        end_token = self.num_tokens - 1
        self.end_token = end_token
        self.pos_emb_inds = torch.cat((cat_code_ind, skeleton_pos_inds, end_code_inds))


    def __len__(self):
        return len(self.skeletons)


    def __getitem__(self, ind):
        # grab a chunk of (block_size + 1) characters from the data
        #cloud_skeleton = self.skeletons[ind]
        cloud_center = self.skeletons[ind]
        #print(cloud_center.shape)

        if self.select_random:
            cur_ind = np.random.randint(0, len(cloud_center), size=1)[0]
        else:
            cur_ind = 0

        end_token = torch.LongTensor([self.end_token])
        skeleton_tokens = cloud_center[cur_ind].reshape(-1)
        dix = np.concatenate((skeleton_tokens, end_token))
        #print(dix.shape)

        category_token = np.array([self.num_base_tokens + self.categories_dict[self.categories[ind]]])
        dix = np.concatenate((category_token, dix))
        #print(dix.shape)
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)

        return x, y

    def vectorize(self, x, f):
        return np.vectorize(f)(x)

    def sort_clouds_in_4d(self, input_array):
        assert len(input_array.shape) == 4

        all_clouds = []
        print('debug')
        for i in range(len(input_array)):
            sorted_clouds = []
            for j in range(input_array.shape[1]):
                cur_skeleton = input_array[i, j]
                lexsort_inds = np.lexsort(cur_skeleton[:, [2, 0, 1]].T)
                sorted_clouds += [cur_skeleton[lexsort_inds]]
            sorted_clouds = np.stack(sorted_clouds, axis=0)
            all_clouds += [sorted_clouds]

        all_clouds = np.stack(all_clouds, axis=0)

        return all_clouds


    def sort_clouds_in_4d_fast(self, input_array, sort_device=None):
        assert len(input_array.shape) == 4

        B, K, N, D = input_array.shape
        flat_array = input_array.reshape(-1, N, D)
        mult_array = np.array([[[10, 100, 1]]])
        sort_keys = (mult_array * flat_array).sum(axis=-1)

        torch_keys = torch.FloatTensor(sort_keys).to(sort_device)
        sort_ind = torch.argsort(torch_keys, axis=-1).detach().cpu().numpy()

        sort_flat_array = np.take_along_axis(flat_array, sort_ind[:, :, None], axis=1)
        sort_flat_array = sort_flat_array.reshape(B, K, N, D)

        return sort_flat_array


