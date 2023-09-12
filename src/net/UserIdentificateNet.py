# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class UserIdentificateNet(nn.Module):
    def __init__(self, hidden_size, n_user, negative_prop, device):
        super(UserIdentificateNet, self).__init__()
        self.hidden_size = hidden_size
        self.n_user = n_user
        self.negative_prop = negative_prop
        self.device = device
        self.encode = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size, bias=True), nn.ReLU(inplace=True),
                                    nn.Linear(self.hidden_size, self.hidden_size, bias=True))
        # self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        # self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        
    def forward(self, user_ids, sess_embedding, gnn_sess_embedding):
        
        unique_user = torch.unique(user_ids).cpu()
        
        positive_index = torch.tensor([]).long()
        for user_id in unique_user:
            indices = torch.nonzero(user_ids==user_id).cpu()
            if indices.shape[0]>1:
                id_combinations = torch.combinations(indices.squeeze())
                positive_index = torch.cat([positive_index, id_combinations], 0)
        positive = torch.cat((positive_index, torch.ones(positive_index.shape[0],1).long()), dim=1)

        negative_samples = int(positive_index.shape[0]+1 * self.negative_prop)
        negative_index = torch.tensor([]).long()
        while negative_index.shape[0] < negative_samples and unique_user.shape[0]>1:
            random_ids = np.random.choice(unique_user, 2, replace=False)
            # Get the indices where the selected user IDs match
            indices_1 = torch.nonzero(user_ids == random_ids[0]).squeeze().cpu()
            indices_2 = torch.nonzero(user_ids == random_ids[1]).squeeze().cpu()
            id_combinations = torch.cartesian_prod(indices_1.squeeze().view(-1), indices_2.squeeze().view(-1))
            negative_index = torch.cat([negative_index, id_combinations], 0)
        negative_index = negative_index[:negative_samples]
        negative = torch.cat((negative_index, torch.zeros(negative_index.shape[0],1).long()), dim=1)

        if positive.shape[0]!=0 and negative.shape[0]!=0:
            total = torch.cat((positive,negative),0)
        elif negative.shape[0]==0:
            total = positive
        elif positive.shape[0]==0:
            total = negative
        else:
            raise AssertionError('positive and negative tensors are empty')
            
        idx = torch.randperm(total.shape[0])
        total = total[idx].view(total.size()).long().to(self.device)
        
        first = self.encode(sess_embedding[total[:,0]])
        second = self.encode(gnn_sess_embedding[total[:,1]])
        tgt = total[:,2].reshape(-1,1).float()
        
        out = torch.mm(first, second.transpose(1,0)).diag().view(-1,1)
        # out = self.linear(torch.cat([first,second], 1))
        
        return out, tgt