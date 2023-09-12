# -*- coding: utf-8 -*-
"""
Created on 13/5/2023
@author: Jisoo Cha
"""

import collections
import torch
import torch.nn.functional as F
from torch_geometric.data import Data


def build_usergraph(hidden, user_embed, user_ids, sess_ids, batch_size, threshold, device):
    '''
    Input:
        hidden : input hidden embeddings of session nodes, size = (input_size, hidden_size)
        user_ids : data's user ids
        sess_ids : data's sess ids
    Output:
        user_graph : torch_geometric.data.Data
    '''
    input_size = hidden.shape[0]
    sorted_user_indices = torch.argsort(user_ids)
    sorted_user = user_ids[sorted_user_indices]
    group_chunks = torch.bincount(sorted_user)
    sess_splitted = torch.split(sorted_user_indices, tuple(group_chunks.cpu().numpy()))
    user_chunk = tuple(indices[torch.argsort(sess_ids[indices])] for indices in sess_splitted) #user id starts at 1
    
    user_sim = torch.softmax(torch.mm(user_embed, user_embed.T), 1).detach().cpu().numpy()
    pair = {}
    for i in range(user_sim.shape[0]):
        for j in range(user_sim.shape[1]):
            if i<j:
                edge = str(user_ids[i].item() + batch_size)+'-'+str(user_ids[j].item() + batch_size)
                weight = user_sim[i][j]
                if weight > threshold:
                    pair[edge] = weight
    
    user_src = [int(i.split('-')[0]) for i in list(pair)] + [int(i.split('-')[1]) for i in list(pair)]
    user_tgt = [int(i.split('-')[1]) for i in list(pair)] + [int(i.split('-')[0]) for i in list(pair)]
    weight = list(pair.values()) + list(pair.values())
    
    src = []
    tgt = []
    star = []
    sat = []
    for i,chunk in enumerate(user_chunk):
        if chunk.shape[0]:
            src.extend(chunk[:-1].tolist())
            tgt.extend(chunk[1:].tolist())
            star.extend([i+batch_size]*chunk.shape[0])
            sat.extend(chunk.tolist())
    
    # total_src = src + user_src
    # total_tgt = tgt + user_tgt
    padded_embed = F.pad(hidden, (0,0,0,batch_size-input_size), 'constant', 0)
    
#     count = collections.Counter(src)
#     out_degree_inv = [1 / count[i] for i in src]

#     count = collections.Counter(tgt)
#     in_degree_inv = [1 / count[i] for i in tgt]

#     in_degree_inv = torch.tensor(in_degree_inv, dtype=torch.float)
#     out_degree_inv = torch.tensor(out_degree_inv, dtype=torch.float)
    
#     pair = {}
#     sur_senders = src[:]
#     sur_receivers = tgt[:]
#     i = 0
#     for sender, receiver in zip(sur_senders, sur_receivers):
#         if str(sender) + '-' + str(receiver) in pair:
#             pair[str(sender) + '-' + str(receiver)] += 1
#             del src[i]
#             del tgt[i]
#         else:
#             pair[str(sender) + '-' + str(receiver)] = 1
#             i += 1
            
#     edge_count = [pair[str(src[i]) + '-' + str(tgt[i])] for i in range(len(src))]
#     edge_count = torch.tensor(edge_count, dtype=torch.float)
    
    user_edge_index = torch.tensor([user_src, user_tgt], dtype=torch.long).to(device)
    user2sess_index = torch.tensor([star, sat], dtype=torch.long).to(device)
    sess_edge_index = torch.tensor([src, tgt], dtype=torch.long).to(device)
    # stargnn_index = torch.tensor([sat, star], dtype=torch.long).to(device)
    # usergnn_index = torch.tensor([user_src, user_tgt], dtype=torch.long).to(device)
    user_edge_weight = torch.tensor(weight).to(device)
    # edge_weight = [i.to(device) for i in edge_weight]
    
    return (padded_embed, user_edge_index, user_edge_weight, user2sess_index, sess_edge_index)
