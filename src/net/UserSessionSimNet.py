# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_softmax

class UserSessionSimNet(nn.Module):
    def __init__(self, hidden_size):
        super(UserSessionSimNet, self).__init__()
        self.hidden_size = hidden_size
        self.linear_one = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.Wq = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.Wk = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.Wv = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        
    def forward(self, sess_embed, user_embed, user_ids):
        
        query = torch.mm(user_embed, self.Wq)
        key = torch.mm(sess_embed, self.Wk)
        value = torch.mm(user_embed, self.Wv)
        
        similarity = torch.mm(query, key.T).diag()
        sim_sm = scatter_softmax(similarity, user_ids)
        weighted_user = user_embed * sim_sm[:,None]
        weighted_sum = scatter_add(weighted_user, user_ids, dim=0)
        
        user_concat = torch.cat((value, weighted_sum[user_ids]), dim=1)
        hidden = self.linear_one(user_concat)
        hidden = F.relu(hidden)
        
        return hidden