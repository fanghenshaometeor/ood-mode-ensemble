import os
import sys
sys.path.append("..")
sys.path.append("../model")
import time
import torch
import faiss
import numpy as np

import sys
import argparse

from utils import setup_seed
from utils import Logger

import metrics

# ======== options ==============
parser = argparse.ArgumentParser(description='Evaluation on clean samples')
# -------- file param. --------------
parser.add_argument('--data_dir',type=str,default='./data/CIFAR10/',help='data directory')
parser.add_argument('--logs_dir',type=str,default='./logs/',help='logs directory')
parser.add_argument('--cache_dir',type=str,default='./cache/',help='logs directory')
parser.add_argument('--dataset',type=str,default='CIFAR10',help='data set name')
parser.add_argument('--model_path',type=str,default='./save/CIFAR10-VGG.pth',help='saved model path')
# -------- hyper param. --------
parser.add_argument('--arch',type=str,default='vgg16',help='model architecture')
parser.add_argument('--seed',type=int,default=0,help='random seeds')
parser.add_argument('--train_seed',type=int,default=1000,help='random seeds')
parser.add_argument('--num_classes',type=int,default=10,help='num of classes')
parser.add_argument('--batch_size',type=int,default=256,help='batch size for training (default: 256)')    
# -------- ood param. --------
parser.add_argument('--score', choices=['MSP', 'ODIN', 'Energy', 'Mahalanobis', 'GradNorm','RankFeat','kNN'], default='MSP')
parser.add_argument('--in_data', choices=['CIFAR10', 'ImageNet'], default='CIFAR10')
parser.add_argument('--in_datadir', type=str, help='in data dir')
parser.add_argument('--out_data', choices=['SVHN','LSUN','iSUN','Texture','places365'], default='SVHN')
parser.add_argument('--out_datadir', type=str, help='out data dir')
parser.add_argument('--out_datasets',default=['SVHN','LSUN','iSUN','Texture','places365'], nargs="*", type=str)
args = parser.parse_args()

args.dataset = args.in_data
hyperparam='seed-%d'%args.train_seed
args.logs_path = os.path.join(args.logs_dir,args.dataset,args.arch,'eval',hyperparam+'-ood.log')
args.cache_path = os.path.join(args.cache_dir,args.dataset,args.arch,hyperparam)
sys.stdout = Logger(filename=args.logs_path,stream=sys.stdout)

setup_seed(args.seed)

cache_name = os.path.join(args.cache_path, "train_in.npy")
feat_log = np.load(cache_name, allow_pickle=True)
feat_log = feat_log.T.astype(np.float32)

cache_name = os.path.join(args.cache_path, "val_in.npy")
feat_log_val = np.load(cache_name, allow_pickle=True)
feat_log_val = feat_log_val.T.astype(np.float32)

ood_feat_log_all = {}
for ood_dataset in args.out_datasets:
    cache_name = os.path.join(args.cache_path, f"{ood_dataset}_out.npy")
    ood_feat_log = np.load(cache_name, allow_pickle=True)
    ood_feat_log = ood_feat_log.T.astype(np.float32)
    ood_feat_log_all[ood_dataset] = ood_feat_log

normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
prepos_feat = lambda x: np.ascontiguousarray(normalizer(x))# Last Layer only

ftrain = prepos_feat(feat_log)
ftest = prepos_feat(feat_log_val)
food_all = {}
for ood_dataset in args.out_datasets:
    food_all[ood_dataset] = prepos_feat(ood_feat_log_all[ood_dataset])

#################### KNN score OOD detection #################

index = faiss.IndexFlatL2(ftrain.shape[1])
index.add(ftrain)
for K in [50]:

    D, _ = index.search(ftest, K)
    scores_in = -D[:,-1]
    all_results = []
    all_score_ood = []
    for ood_dataset, food in food_all.items():
        D, _ = index.search(food, K)
        scores_ood_test = -D[:,-1]
        all_score_ood.extend(scores_ood_test)
        results = metrics.cal_metric(scores_in, scores_ood_test)
        all_results.append(results)

    metrics.print_all_results(all_results, args.out_datasets, f'KNN k={K}')
    print()


