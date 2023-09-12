    # -*- coding: utf-8 -*-
"""
Created on 4/4/2019
@author: RuihongQiu

Edited 30/5/2023
@Editor: Jisoo Cha
"""

import os
import argparse
import logging
import time
import torch
import random
import numpy as np
from tqdm import tqdm
from dataset import MultiSessionsGraph
from torch_geometric.data import DataLoader
from utils.construct_usergraph import construct_usergraph
from model import GNNModel
from train import train_test
from tensorboardX import SummaryWriter


# Logger configuration
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(message)s')


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name')
parser.add_argument('--dataset_path', default='/home/jisoo/data/dataset', help='dataset path')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=128, help='hidden state size')
parser.add_argument('--epoch', type=int, default=15, help='the number of epochs to train for')
parser.add_argument('--lam', type=float, default=0.1, help='lambda of unsupervised loss')
parser.add_argument('--threshold', type=float, default=0, help='threshold of user similarity')
parser.add_argument('--negative_prop', type=float, default=2, help='negative_prop for contrastive loss')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--u2uedge_k', type=float, default=0.1, help='u2uedge_k')
parser.add_argument('--top_k', type=str, default='1,3,5,10', help='top K indicator for evaluation')
parser.add_argument('--model_save', type=bool, default=False, help='model save or not. default location: ../models')
parser.add_argument('--gpu_use', type=str, default='0', help='index of gpu to use')
parser.add_argument('--comment', type=str, default='', help='logfolder comment')
opt = parser.parse_args()
logging.warning(opt)


def set_seed(seed=42):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main():
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_use 
    
    opt.device = torch.device('cuda:'+opt.gpu_use if torch.cuda.is_available() else 'cpu')
    set_seed()

    if opt.dataset == 'appusage':
        n_node = 2301
        n_user = 34
    elif opt.dataset == 'kobaco_16' or opt.dataset == 'kobaco_64':
        n_node = 17596
        n_user = 4847
    else:
        raise AssertionError('dataset not supported')
        
    log_dir = '../log/' + str(opt.dataset) + '/' + time.strftime("%m%d", time.localtime()) + str(opt) + opt.comment
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.warning('logging to {}'.format(log_dir))
    writer = SummaryWriter(log_dir)

    
    i2i_data, i2u_data, u2i_data = construct_usergraph(os.path.join(opt.dataset_path, opt.dataset), n_node+1, n_user+1)
    
    train_dataset = MultiSessionsGraph(os.path.join(opt.dataset_path,opt.dataset), phrase='train')
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    test_dataset = MultiSessionsGraph(os.path.join(opt.dataset_path,opt.dataset), phrase='test')
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)
    
    print(f'# of items : {n_node}')
    print(f'# of users : {n_user}')
    print(f'# of train set : {len(train_dataset)}')
    print(f'# of test set : {len(test_dataset)}')
    
    model = GNNModel(opt=opt, n_node=n_node+1, n_user=n_user+1, usergraph_i2i=i2i_data, usergraph_i2u=i2u_data, usergraph_u2i=u2i_data)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

    logging.warning(model)
    train_test(opt, model, train_loader, test_loader, writer, optimizer, scheduler)
    


if __name__ == '__main__':
    main()
