# -*- coding: utf-8 -*-
"""
Created on 4/4/2019
@author: RuihongQiu

Edited 30/5/2023
@editor: Jisoo Cha

"""


import math
import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Dropout
import torch.nn.functional as F
from torch_geometric.nn import Sequential
from torch_geometric.nn.conv import SAGEConv
from net import InOutGGNN, UserIdentificateNet, UserSessionSimNet
from torch_geometric.data import Data


class Item2SessionEmbedding(nn.Module):
    def __init__(self, hidden_size):
        super(Item2SessionEmbedding, self).__init__()
        self.hidden_size = hidden_size
        self.W_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.q = nn.Linear(self.hidden_size, 1)
        self.W_3 = nn.Linear(2 * self.hidden_size, self.hidden_size)
        

    def forward(self, node_embedding, batch, num_count):
        sections = torch.bincount(batch)
        v_i = torch.split(node_embedding, tuple(sections.cpu().numpy()))    # split whole x back into graphs G_i
        v_n_repeat = tuple(nodes[-1].view(1, -1).repeat(nodes.shape[0], 1) for nodes in v_i)    # repeat |V|_i times for the last node embedding

        # Eq(6)
        alpha = self.q(torch.sigmoid(self.W_1(torch.cat(v_n_repeat, dim=0)) + self.W_2(node_embedding)))    # |V|_i * 1
        s_g_whole = num_count.view(-1, 1) * alpha * node_embedding    # |V|_i * hidden_size
        s_g_split = torch.split(s_g_whole, tuple(sections.cpu().numpy()))    # split whole s_g into graphs G_i
        s_g = tuple(torch.sum(embeddings, dim=0).view(1, -1) for embeddings in s_g_split)

        # Eq(7)
        v_n = tuple(nodes[-1].view(1, -1) for nodes in v_i)
        s_h = self.W_3(torch.cat((torch.cat(v_n, dim=0), torch.cat(s_g, dim=0)), dim=1))
        
        return s_h
    
class GNNModel(nn.Module):
    """
    Args:
        hidden_size: the number of units in a hidden layer.
        n_node: the number of items in the whole item set for embedding layer.
    """
    def __init__(self, opt, n_node, n_user, usergraph_i2i, usergraph_i2u, usergraph_u2i):
        
        super(GNNModel, self).__init__()
        
        self.hidden_size = opt.hidden_size
        self.n_node, self.n_user = n_node, n_user
        self.negative_prop = opt.negative_prop
        self.device = opt.device
        self.u2uedge_k = opt.u2uedge_k
        self.usergraph_i2i = usergraph_i2i.to(self.device)
        self.usergraph_i2u = usergraph_i2u.to(self.device)
        self.usergraph_u2i = usergraph_u2i.to(self.device)
        self.embedding = nn.Embedding(self.n_node+self.n_user, self.hidden_size)
        # matrix transforms user embedding to metric space
        self.user_weight = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        
        self.ggnn = InOutGGNN(self.hidden_size, num_layers=1)
        self.gnn_i2i = SAGEConv(self.hidden_size, self.hidden_size, 'mean')
        self.gnn_i2u = SAGEConv(self.hidden_size, self.hidden_size, 'mean')
        self.gnn_u2i = SAGEConv(self.hidden_size, self.hidden_size, 'mean')
        if self.u2uedge_k>0:
            self.gnn_u2u = SAGEConv(self.hidden_size, self.hidden_size, 'mean')
        self.user_weighted_sum = UserSessionSimNet(self.hidden_size)
        self.user_cls = UserIdentificateNet(hidden_size=self.hidden_size, 
                                            n_user=self.n_user, 
                                            negative_prop=self.negative_prop, 
                                            device=self.device)
        self.i2s_sess = Item2SessionEmbedding(self.hidden_size)
        self.i2s_gnnsess = Item2SessionEmbedding(self.hidden_size)
        
        self.linear = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.sess_user_concat = nn.Linear(self.hidden_size * 2, 1, bias=True)
        self.loss_function = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.reset_parameters()
        self.to(self.device)
        
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def user_sess_sim(self, sess_embed, user_embed):
        embed = torch.cat([sess_embed, user_embed], 1)
        embed = self.sess_user_concat(embed)
        return torch.sigmoid(embed)
    
    def user_sim(self, user_embed, user_ids, threshold):
        
        user_sim = torch.sigmoid(torch.mm(user_embed, user_embed.T) / torch.sqrt(torch.tensor(self.hidden_size)))
        filtered = torch.where(user_sim > self.threshold)
        edge_index = torch.stack(filtered, 0)
        dic = {i:a.item() for i,a in enumerate(user_ids)}
        edge_index = edge_index.cpu().apply_(dic.get)
        
        return edge_index.to(self.device)
        
    def forward(self, data, usergraph_u2u):
        x, edge_index, batch, edge_count, in_degree_inv, out_degree_inv, sequence, num_count, user_ids, sess_ids = \
            data.x - 1, data.edge_index, data.batch, data.edge_count, data.in_degree_inv, data.out_degree_inv,\
            data.sequence, data.num_count, data.user_ids, data.sess_ids


        # reindex user ids
        user_ids = user_ids + self.n_node
        
        # sessiongraph part
        item_embedding = self.embedding(x).squeeze()
        hidden = self.ggnn(item_embedding, edge_index, [edge_count * in_degree_inv, edge_count * out_degree_inv])
        sess_embed = self.i2s_sess(hidden, batch, num_count)
        
        # reconstruct user embedding
        user_embed = self.embedding(user_ids).squeeze()
        weighted_user_embed = self.user_weighted_sum(sess_embed, user_embed, user_ids)
        
        # usergraph part
        glob_embed = self.embedding.weight
        if self.u2uedge_k>0:
            gnn_out = self.gnn_u2u(glob_embed, usergraph_u2u.edge_index)
        gnn_out = self.gnn_i2u(glob_embed, self.usergraph_i2u.edge_index)
        gnn_out = self.gnn_i2i(gnn_out, self.usergraph_i2i.edge_index)
        
        gnn_item_embed = (item_embedding + gnn_out[x].squeeze()) / 2
        gnn_user_embed = (weighted_user_embed + gnn_out[user_ids].squeeze()) / 2
        gnn_sess_embed = self.i2s_gnnsess(gnn_item_embed, batch, num_count)
                
        # combine
        alpha = self.user_sess_sim(gnn_sess_embed, gnn_user_embed)
        hidden = sess_embed + (alpha * gnn_sess_embed + (1 - alpha) * gnn_user_embed)
        scores = torch.mm(hidden, self.embedding.weight[:self.n_node].transpose(1, 0))
        out, tgt = self.user_cls(user_ids, user_embed, gnn_user_embed)
  
        return scores, out, tgt