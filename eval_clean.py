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
parser.add_argument('--model_path',type=str,default='./save/CIFAR10-VGG.pth',help='saved model path')
parser.add_argument('--arch',type=str,default='vgg16',help='model architecture')
parser.add_argument('--seed',type=int,default=0,help='random seeds')
parser.add_argument('--batch_size',type=int,default=256,help='batch size for training (default: 256)')    

args = parser.parse_args()

# ======== log writer init. ========
hyperparam=os.path.split(os.path.split(args.model_path)[-2])[-1]
if not os.path.exists(os.path.join(args.logs_dir,args.dataset,args.arch,'eval')):
    os.makedirs(os.path.join(args.logs_dir,args.dataset,args.arch,'eval'))
args.logs_path = os.path.join(args.logs_dir,args.dataset,args.arch,'eval',hyperparam+'-clean.log')
sys.stdout = Logger(filename=args.logs_path,stream=sys.stdout)

# -------- main function
def main():

    # ======== fix random seed ========
    setup_seed(args.seed)
    
    # ======== get data set =============
    trainloader, testloader = get_datasets(args)
    print('-------- DATA INFOMATION --------')
    print('---- dataset: '+args.dataset)

    # ======== load network ========
    checkpoint = torch.load(args.model_path, map_location=torch.device("cpu"))
    net = get_model(args).cuda()
    net.load_state_dict(checkpoint['state_dict'])
    print('-------- MODEL INFORMATION --------')
    print('---- arch.: '+args.arch)
    print('---- saved path: '+args.model_path)
    print('---- inf. seed.: '+str(args.seed))

    # ======== evaluation on clean ========
    print('Validating...')
    if args.dataset == 'ImageNet':
        acc_te = val(net, testloader)
        print('     test acc. = %.2f.'%(acc_te.avg))
    else:
        acc_tr, acc_te = val(net, trainloader), val(net, testloader)
        print('     train/test acc. = %.3f/%.2f.'%(acc_tr.avg, acc_te.avg))


    print("Finished.")



    return

def val(net, dataloader):
    
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
            logits = net(images).detach().float()
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