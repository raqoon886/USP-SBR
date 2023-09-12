# -*- coding: utf-8 -*-
"""
Created on 26/5/2023
@author: Jisoo Cha
"""

import os
import pickle
import math
import torch
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.transforms import RemoveDuplicatedEdges


def construct_usergraph(dataset_path, item_num, user_num, drop_duplicates=True):
    '''
    Input:
        dataset_path : path of dataset.
        item_num : n of items.
        user_num : n of users.
    Output:
        tuple of user_graph : torch_geometric.data.Data
    '''
    data = pickle.load(open(os.path.join(dataset_path, 'raw', 'train.pkl'), 'rb'))
    train_data = {}
    print('processing...')
    for idx,sess in enumerate(tqdm(data['session_data'])):
        if data['user_idx'][idx] not in train_data:
            train_data[data['user_idx'][idx]] = []
        train_data[data['user_idx'][idx]].append(sess)

    src = []
    tgt = []
    sess_u = []
    u_sess = []
    for user in train_data:
        for sess in train_data[user]:
            src += sess[:-1]
            tgt += sess[1:]
            sess_u += sess
            u_sess += [user]*len(sess)
    
    # reindex user ids
    u_sess = [u+item_num for u in u_sess]        
        
    i2i_edge = torch.tensor([src, tgt], dtype=torch.long)
    i2u_edge = torch.tensor([sess_u, u_sess], dtype=torch.long)
    u2i_edge = torch.tensor([u_sess, sess_u], dtype=torch.long)
    i2i_data = Data(num_nodes=item_num+user_num, edge_index = i2i_edge)
    i2u_data = Data(num_nodes=item_num+user_num, edge_index = i2u_edge)
    u2i_data = Data(num_nodes=item_num+user_num, edge_index = u2i_edge)
    
    if drop_duplicates:
        transform = RemoveDuplicatedEdges()
        i2i_data = transform(i2i_data)
        i2u_data = transform(i2u_data)
        u2i_data = transform(u2i_data)
    
    return i2i_data, i2u_data, u2i_data

def rank_u2uedge(model, u2uedge_k: float):

    # compute user similarity and rank them by given parameter k

    n_node = model.n_node
    n_user = model.n_user
    user_embed = model.embedding.weight[model.n_node:].cpu().detach()
    user_embed = user_embed / user_embed.norm(dim=1)[:, None]
    res = torch.mm(user_embed, user_embed.transpose(0,1))
    res = torch.abs(res.fill_diagonal_(0))
    res = torch.triu(res)
    top_indices = torch.topk(res.flatten(), u2uedge_k * n_user * n_user).indices
    indices = [return_elements(n_user, idx) for idx in top_indices]
    indices = torch.tensor(indices).T + n_node

    return Data(num_nodes=n_node+n_user, edge_index = indices)
    
def return_elements(n_rows, i):
    # flattened index to matrix index
    return (i//n_rows).numpy().item(), (i%n_rows).numpy().item()