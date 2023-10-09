from __future__ import print_function

import torch
import torch.optim as optim
import torch.nn.functional as F 

from torch.utils.tensorboard import SummaryWriter

import os
import sys
import time
import argparse
import numpy as np

from utils import setup_seed
from utils import get_datasets, get_model
from utils import Logger
from utils import AverageMeter, accuracy

# ======== fix data type ========
torch.set_default_tensor_type(torch.FloatTensor)

# ======== options ==============
parser = argparse.ArgumentParser(description='Training')
# -------- file param. --------------
parser.add_argument('--data_dir',type=str,default='./data/CIFAR10/',help='data directory')
parser.add_argument('--logs_dir',type=str,default='./logs/',help='logs directory')
parser.add_argument('--save_dir',type=str,default='./save/',help='model saving directory')
parser.add_argument('--runs_dir',type=str,default='./runs/',help='tensorboard saving directory')
parser.add_argument('--dataset',type=str,default='CIFAR10',help='data set name')
# -------- training param. ----------
parser.add_argument('--batch_size',type=int,default=256,help='batch size for training (default: 256)')    
parser.add_argument('--lr_init',type=float,default=0.1,help='init. learning rate (default: 0.1)')
parser.add_argument('--wd',type=float,default=1e-4,help='weight decay')
parser.add_argument('--epochs',type=int,default=150,help='number of epochs to train')
parser.add_argument('--save_freq',type=int,default=150,help='model save frequency')
parser.add_argument('--arch',type=str,default='vgg16',help='model architecture')
parser.add_argument('--seed',type=int,default=0,help='random seeds')

args = parser.parse_args()

# ======== log writer init. ========
hyperparam='seed-'+str(args.seed)
writer = SummaryWriter(os.path.join(args.runs_dir,args.dataset,args.arch,hyperparam+'/'))
if not os.path.exists(os.path.join(args.save_dir,args.dataset,args.arch,hyperparam)):
    os.makedirs(os.path.join(args.save_dir,args.dataset,args.arch,hyperparam))
if not os.path.exists(os.path.join(args.logs_dir,args.dataset,args.arch,'train')):
    os.makedirs(os.path.join(args.logs_dir,args.dataset,args.arch,'train'))
args.save_path = os.path.join(args.save_dir,args.dataset,args.arch,hyperparam)
args.logs_path = os.path.join(args.logs_dir,args.dataset,args.arch,'train',hyperparam+'-train.log')
sys.stdout = Logger(filename=args.logs_path,stream=sys.stdout)



# -------- main function
def main():

    # ======== fix random seed ========
    setup_seed(args.seed)
    
    # ======== get data set =============
    trainloader, testloader = get_datasets(args)
    print('-------- DATA INFOMATION --------')
    print('---- dataset: '+args.dataset)

    # ======== initialize net
    net = get_model(args).cuda()
    print('-------- MODEL INFORMATION --------')
    print('---- arch.: '+args.arch)

    # ======== initialize optimizer
    optimizer = optim.SGD(net.parameters(), lr=args.lr_init, momentum=0.9, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    print('-------- START TRAINING --------')
    for epoch in range(1, args.epochs+1):

        # -------- train
        print('Training(%d/%d)...'%(epoch, args.epochs))
        train_epoch(net, trainloader, optimizer, epoch)
        scheduler.step()

        # -------- validation
        print('Validating...')
        acc_te = val(net, testloader)
        writer.add_scalar('valacc', acc_te.avg, epoch)
        print('     Current test acc. = %f.'%acc_te.avg)

        # -------- save model & print info
        if (epoch == 1 or epoch % args.save_freq == 0 or epoch == args.epochs):
            checkpoint = {'state_dict': net.state_dict()}
            args.model_path = 'epoch%d'%epoch+'.pth'
            torch.save(checkpoint, os.path.join(args.save_path,args.model_path))

        print('Current training %s on data set %s.'%(args.arch, args.dataset))
        print('===========================================')
    print('Finished training: ', args.save_path)

    
    return

def train_epoch(net, trainloader, optimizer, epoch):
    net.train()

    batch_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for batch_idx, (b_data, b_label) in enumerate(trainloader):
        
        # -------- move to gpu
        b_data, b_label = b_data.cuda(), b_label.cuda()

        logits = net(b_data)
        loss_ce = F.cross_entropy(logits, b_label)
        
        # -------- backprop. & update
        optimizer.zero_grad()
        loss_ce.backward()
        optimizer.step()

        # -------- record & print in termial
        losses.update(loss_ce.float().item(), b_data.size(0))
        batch_time.update(time.time()-end)
        end = time.time()

    writer.add_scalar('loss-ce', losses.avg, epoch)
    print('     Epoch %d/%d costs %fs.'%(epoch, args.epochs, batch_time.sum))
    print('     CE loss = %f.'%losses.avg)

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