# -*- coding: utf-8 -*-
"""
Created on 5/4/2019
@author: RuihongQiu

Edited 30/5/2023
@Editor: Jisoo Cha
"""

import os
import time
from tqdm import tqdm
import numpy as np
import logging
import torch
from termcolor import colored

import os
import pickle
import math
import torch
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.transforms import RemoveDuplicatedEdges


def print_score(hit, mrr, top_k, color=False):
    for k in top_k:
        tmp_str = 'HIT@{} : {:.4f}, MRR@{} : {:.4f}'.format(k, hit[k], k, mrr[k])
        if color:
            tmp_str = colored('BEST ' + tmp_str, 'red', attrs=['bold'])
        print(tmp_str)

        
def train(model, loader, epoch, lam, writer, optimizer, scheduler, device, u2u_edge):
    
    model.train()
    ce_mean_loss = 0.0
    cont_mean_loss = 0.0
    updates_per_epoch = len(loader)
    
    start = time.time()
    for i, batch in enumerate(loader):

        optimizer.zero_grad()
        scores, out, tgt = model(batch.to(device), u2u_edge)
        targets = batch.y - 1
        ce_loss = (1 - lam) * model.loss_function(scores, targets)
        cont_loss = lam * model.bce_loss(out, tgt)
        loss = ce_loss + cont_loss

        loss.backward()
        optimizer.step()
        writer.add_scalar('loss/train_batch_loss', loss.item(), epoch * updates_per_epoch + i)

        ce_mean_loss += ce_loss / batch.num_graphs
        cont_mean_loss += cont_loss / batch.num_graphs

    duration = time.time() - start
    print('training time : {:.2f}s'.format(duration))
    print('TRAIN recommendation loss : {:.3f}, contrastive loss : {:.6f}'.format(ce_mean_loss.item(), cont_mean_loss.item()))
    
    writer.add_scalar('loss/train_ce_loss', ce_mean_loss.item(), epoch)
    writer.add_scalar('loss/train_cont_loss', cont_mean_loss.item(), epoch)
    scheduler.step()
    
    return duration

def test(model, loader, epoch, lam, writer, best_hit, best_mrr, top_k, device, u2u_edge):
    
    model.eval()
    ce_mean_loss = 0.0
    cont_mean_loss = 0.0
    hit, mrr = {}, {}
    for k in top_k:
        hit[k] = []
        mrr[k] = []
    
    start = time.time()
    for i, batch in enumerate(loader):
            
        with torch.no_grad():
            scores, out, tgt = model(batch.to(device), u2u_edge)
            targets = batch.y - 1
            ce_loss = (1 - lam) * model.loss_function(scores, targets)
            cont_loss = lam * model.bce_loss(out, tgt)
            for k in top_k:
                sub_scores = scores.topk(k)[1]    # batch * top_k
                for score, target in zip(sub_scores.detach().cpu().numpy(), targets.detach().cpu().numpy()):
                    hit[k].append(np.isin(target, score))
                    if len(np.where(score == target)[0]) == 0:
                        mrr[k].append(0)
                    else:
                        mrr[k].append(1 / (np.where(score == target)[0][0] + 1))

            ce_mean_loss += ce_loss / batch.num_graphs
            cont_mean_loss += cont_loss / batch.num_graphs
            
    duration = time.time() - start
    print('evaluation time : {:.2f}s'.format(duration))
    print('TEST recommendation loss : {:.3f}, contrastive loss : {:.6f}'.format(ce_mean_loss.item(), cont_mean_loss.item()))
    writer.add_scalar('loss/test_ce_loss', ce_mean_loss.item(), epoch)
    writer.add_scalar('loss/test_cont_loss', cont_mean_loss.item(), epoch)

    for k in top_k:
        hit_k = np.mean(hit[k]) * 100
        mrr_k = np.mean(mrr[k]) * 100
        writer.add_scalar(f'index/hit_{k}', hit_k, epoch)
        writer.add_scalar(f'index/mrr_{k}', mrr_k, epoch)
        best_hit[k] = max(best_hit[k], hit_k)
        best_mrr[k] = max(best_mrr[k], mrr_k)
        hit[k] = hit_k
        mrr[k] = mrr_k
    
    print_score(hit, mrr, top_k)
    
    return duration, best_hit, best_mrr
    
    
def train_test(opt, model, train_loader, test_loader, writer, optimizer, scheduler):
    
    device = opt.device
    top_k = list(map(int, opt.top_k.split(',')))
    best_hit, best_mrr = {}, {}
    for k in top_k:
        best_hit[k] = 0
        best_mrr[k] = 0
    train_times = []
    test_times = []
    
    start = time.time()
    for epoch in range(opt.epoch):
        print(f'Epoch [{epoch+1}/{opt.epoch}]')
        # compute user-user similarity ranking
        if opt.u2uedge_k>0:
            usergraph_u2u = rank_u2uedge(model, opt.u2uedge_k).to(device)
        else:
            usergraph_u2u = None

        # training
        train_time = train(model, train_loader, epoch, opt.lam, writer, optimizer, scheduler, device, usergraph_u2u)
        train_times.append(train_time)
        # evaluating
        test_time, best_hit, best_mrr = test(model, test_loader, epoch, opt.lam, writer, best_hit, best_mrr, top_k, device, usergraph_u2u)
        test_times.append(test_time)
        
        if opt.model_save and (epoch+1)%5==0:
            model_path = os.path.join('../models', opt.dataset, 
                                      f'{time.strftime("%m%d", time.localtime())}_BATCH{opt.batch_size}_lam{opt.lam}_np{opt.negative_prop}_{opt.comment}')
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            torch.save(model.state_dict(), os.path.join(model_path, f'model_epoch{epoch+1}.pt'))
            
        writer.flush()
        print_score(best_hit, best_mrr, top_k, color=True)
        print('current training time : {:.2f}s'.format(time.time()-start))
        print('='*24)
            
    writer.close()
    print(f'avg train times : {np.mean(train_times)}')
    print(f'avg test times : {np.mean(test_times)}')
    print(f'Dataset Name : {opt.dataset}')
    print('end training')

def rank_u2uedge(model, u2uedge_k: float):

    # compute user similarity and rank them by given parameter k

    n_node = model.n_node
    n_user = model.n_user
    user_embed = model.embedding.weight[model.n_node:].cpu().detach()
    user_embed = user_embed / user_embed.norm(dim=1)[:, None]
    res = torch.mm(user_embed, user_embed.transpose(0,1))
    res = torch.abs(res.fill_diagonal_(0))
    res = torch.triu(res)
    top_indices = torch.topk(res.flatten(), int(u2uedge_k * n_user * n_user)).indices
    indices = [return_elements(n_user, idx) for idx in top_indices]
    indices = torch.tensor(indices).T + n_node

    return Data(num_nodes=n_node+n_user, edge_index = indices)
    
def return_elements(n_rows, i):
    # flattened index to matrix index
    return (i//n_rows).numpy().item(), (i%n_rows).numpy().item()