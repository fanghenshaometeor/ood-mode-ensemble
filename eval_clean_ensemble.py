from __future__ import print_function

import torch

import numpy as np
import pandas as pd

import os
import sys
import time
import argparse
import re

from utils import setup_seed
from utils import get_datasets, get_model
from utils import Logger
from utils import AverageMeter, accuracy

# ======== fix data type ========
torch.set_default_tensor_type(torch.FloatTensor)

# ======== options ==============
parser = argparse.ArgumentParser(description='Evaluation on clean samples')
# -------- file param. --------------
parser.add_argument('--data_dir',type=str,default='./data/CIFAR10/',help='data directory')
parser.add_argument('--logs_dir',type=str,default='./logs/',help='logs directory')
parser.add_argument('--dataset',type=str,default='CIFAR10',help='data set name')
parser.add_argument('--model_path',type=str,help='saved model path',nargs='+')
parser.add_argument('--arch',type=str,default='vgg16',help='model architecture')
parser.add_argument('--seed',type=int,default=0,help='random seeds')
parser.add_argument('--batch_size',type=int,default=256,help='batch size for training (default: 256)')    

args = parser.parse_args()

# ======== log writer init. ========
hyperparam = ''
for PATH in args.model_path:
    PATH_name =os.path.split(os.path.split(PATH)[-2])[-1]
    hyperparam = hyperparam + re.findall(r"\d+",PATH_name)[0] + '-'
if not os.path.exists(os.path.join(args.logs_dir,args.dataset,args.arch,'eval-ensemble')):
    os.makedirs(os.path.join(args.logs_dir,args.dataset,args.arch,'eval-ensemble'))
args.logs_path = os.path.join(args.logs_dir,args.dataset,args.arch,'eval-ensemble',hyperparam+'clean.log')
sys.stdout = Logger(filename=args.logs_path,stream=sys.stdout)

# -------- main function
def main():

    # ======== fix random seed ========
    setup_seed(args.seed)
    
    # ======== get data set =============
    trainloader, testloader = get_datasets(args)
    print('-------- DATA INFOMATION --------')
    print('---- dataset: '+args.dataset)

    # ======== load multiple networks ========
    net_ensemble = []
    for PATH in args.model_path:
        checkpoint = torch.load(PATH, map_location=torch.device("cpu"))
        net = get_model(args).cuda()
        net.load_state_dict(checkpoint['state_dict'])
        net_ensemble.append(net)
    print('-------- MODEL INFORMATION --------')
    print('---- arch.: '+args.arch)
    print('---- saved path: ')
    for PATH in args.model_path:
        print('----     '+PATH)
    print('---- inf. seed.: '+str(args.seed))

    # ======== evaluation on clean ========
    print('Validating...')
    acc_tr, acc_te = val_ensemble(net_ensemble, trainloader), val_ensemble(net_ensemble, testloader)
    print('     train/test acc. = %.3f/%.2f.'%(acc_tr.avg, acc_te.avg))


    print("Finished.")



    return

def val_ensemble(net_ensemble, dataloader):
    
    for net in net_ensemble:
        net.eval()

    batch_time = AverageMeter()
    acc = AverageMeter()
    
    end = time.time()
    with torch.no_grad():
        
        # -------- compute the accs.
        for test in dataloader:
            images, labels = test
            images, labels = images.cuda(), labels.cuda()

            # ------- forward 
            logits = 0
            for net in net_ensemble:
                logits = logits + net(images).detach().float()
            prec1 = accuracy(logits.data, labels)[0]
            acc.update(prec1.item(), images.size(0))
            
            # ----
            batch_time.update(time.time()-end)
            end = time.time()

    print('     Validation costs %fs.'%(batch_time.sum))        
    return acc


# ======== startpoint
if __name__ == '__main__':
    main()