#! /usr/bin/env python3

import torch

import os
import sys
sys.path.append("..")
sys.path.append("../model")
import re

import argparse

from utils import setup_seed
from utils import get_model
from utils import Logger
from utils_data import make_id_ood
import numpy as np
import torch.nn.functional as F
import time

# ======== fix data type ========
torch.set_default_tensor_type(torch.FloatTensor)

# ======== options ==============
parser = argparse.ArgumentParser(description='Evaluation on clean samples')
# -------- file param. --------------
parser.add_argument('--data_dir',type=str,default='./data/CIFAR10/',help='data directory')
parser.add_argument('--logs_dir',type=str,default='./logs/',help='logs directory')
parser.add_argument('--cache_dir',type=str,default='./cache/',help='logs directory')
parser.add_argument('--dataset',type=str,default='CIFAR10',help='data set name')
parser.add_argument('--model_path',type=str,help='saved model path',nargs='+')
# -------- hyper param. --------
parser.add_argument('--arch',type=str,default='vgg16',help='model architecture')
parser.add_argument('--seed',type=int,default=0,help='random seeds')
parser.add_argument('--num_classes',type=int,default=10,help='num of classes')
parser.add_argument('--batch_size',type=int,default=32,help='batch size for training (default: 256)')    
# -------- ood param. --------
parser.add_argument('--score', choices=['MSP', 'ODIN', 'Energy', 'Mahalanobis', 'GradNorm','RankFeat','kNN'], default='MSP')
parser.add_argument('--in_data', choices=['CIFAR10', 'ImageNet'], default='CIFAR10')
parser.add_argument('--in_datadir', type=str, help='in data dir')
parser.add_argument('--out_data', choices=['SVHN','LSUN','iSUN','Texture','places365'], default='SVHN')
parser.add_argument('--out_datadir', type=str, help='out data dir')
args = parser.parse_args()

# ======== log writer init. ========
hyperparam = ''
for PATH in args.model_path:
    PATH_name =os.path.split(os.path.split(PATH)[-2])[-1]
    hyperparam = hyperparam + re.findall(r"\d+",PATH_name)[0] + '-'
hyperparam = hyperparam + 'ens'
if not os.path.exists(os.path.join(args.logs_dir,args.dataset,args.arch,'eval')):
    os.makedirs(os.path.join(args.logs_dir,args.dataset,args.arch,'eval'))
if not os.path.exists(os.path.join(args.cache_dir,args.dataset,args.arch,hyperparam)):
    os.makedirs(os.path.join(args.cache_dir,args.dataset,args.arch,hyperparam))
args.logs_path = os.path.join(args.logs_dir,args.dataset,args.arch,'eval',hyperparam+'-ood.log')
args.cache_path = os.path.join(args.cache_dir,args.dataset,args.arch,hyperparam)
# sys.stdout = Logger(filename=args.logs_path,stream=sys.stdout)

setup_seed(args.seed)

_, _, _, testloaderIn, trainloaderIn, out_loader =  make_id_ood(args)

# ======== load multiple networks ========
net_ensemble = []
for PATH in args.model_path:
    checkpoint = torch.load(PATH, map_location=torch.device("cpu"))
    net = get_model(args).cuda()
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    net_ensemble.append(net)
print('-------- MODEL INFORMATION --------')
print('---- arch.: '+args.arch)
print('---- saved path   : ')
for PATH in args.model_path:
    print('----     '+PATH)
print('---- seed in inf. : '+str(args.seed))

batch_size = args.batch_size

FORCE_RUN = False

dummy_input = torch.zeros((1, 3, 32, 32)).cuda()
score, feature_list = net_ensemble[0].feature_list(dummy_input)
featdims = [feature_list[-1].shape[1]]

for split, in_loader in [('train', trainloaderIn), ('val', testloaderIn),]:

    cache_name = os.path.join(args.cache_path, f"{split}_in.npy")
    if FORCE_RUN or not os.path.exists(cache_name):

        feat_log = np.zeros((len(in_loader.dataset), sum(featdims)))

        for batch_idx, (inputs, targets) in enumerate(in_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            start_ind = batch_idx * batch_size
            end_ind = min((batch_idx + 1) * batch_size, len(in_loader.dataset))

            feature = 0
            for net in net_ensemble:
                _, feature_list = net.feature_list(inputs)
                feature = feature + F.adaptive_avg_pool2d(feature_list[-1], 1).squeeze()
            feature = feature / len(net_ensemble)
            out = torch.cat([feature], dim=1)  # <-- adaptive_avg_pool2d, outputsize:(layer_feat, 1)

            feat_log[start_ind:end_ind, :] = out.data.cpu().numpy()
            
            if batch_idx % 100 == 0:
                print(f"{batch_idx}/{len(in_loader)}")
        np.save(cache_name, (feat_log.T))
    else:
        feat_log = np.load(cache_name, allow_pickle=True)
        feat_log = feat_log.T

cache_name = os.path.join(args.cache_path, f"{args.out_data}_out.npy")
if FORCE_RUN or not os.path.exists(cache_name):
    ood_feat_log = np.zeros((len(out_loader.dataset), sum(featdims)))

    for batch_idx, (inputs, _) in enumerate(out_loader):
        inputs = inputs.cuda()
        start_ind = batch_idx * batch_size
        end_ind = min((batch_idx + 1) * batch_size, len(out_loader.dataset))

        feature = 0
        for net in net_ensemble:
            _, feature_list = net.feature_list(inputs)
            feature = feature + F.adaptive_avg_pool2d(feature_list[-1], 1).squeeze()
        feature = feature / len(net_ensemble)
        out = torch.cat([feature], dim=1)  # <-- adaptive_avg_pool2d, outputsize:(layer_feat, 1)


        ood_feat_log[start_ind:end_ind, :] = out.data.cpu().numpy()
        if batch_idx % 100 == 0:
            print(f"{batch_idx}/{len(out_loader)}")
    np.save(cache_name, (ood_feat_log.T))
else:
    ood_feat_log = np.load(cache_name, allow_pickle=True)
    ood_feat_log = ood_feat_log.T
