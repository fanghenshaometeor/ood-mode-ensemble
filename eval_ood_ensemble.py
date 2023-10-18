from __future__ import print_function

import torch
from torch.autograd import Variable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV

import os
import sys
import time
import argparse
import re

from utils import setup_seed
from utils import get_model
from utils import Logger
from utils import AverageMeter, accuracy

from utils_ood import make_id_ood
from utils_ood import iterate_data_msp_ensemble
from utils_ood import iterate_data_energy_ensemble
from utils_ood import iterate_data_odin_ensemble
from utils_ood import iterate_data_mahalanobis_ensemble
from utils_ood import iterate_data_rankfeat_ensemble
from utils_ood import iterate_data_gradnorm_ensemble
from utils_ood import get_measures

# ======== fix data type ========
torch.set_default_tensor_type(torch.FloatTensor)

# ======== options ==============
parser = argparse.ArgumentParser(description='Evaluation on clean samples')
# -------- file param. --------------
parser.add_argument('--data_dir',type=str,default='./data/CIFAR10/',help='data directory')
parser.add_argument('--logs_dir',type=str,default='./logs/',help='logs directory')
parser.add_argument('--dataset',type=str,default='CIFAR10',help='data set name')
parser.add_argument('--model_path',type=str,help='saved model path',nargs='+')
# -------- hyper param. --------
parser.add_argument('--arch',type=str,default='vgg16',help='model architecture')
parser.add_argument('--seed',type=int,default=0,help='random seeds')
parser.add_argument('--num_classes',type=int,default=10,help='num of classes')
parser.add_argument('--batch_size',type=int,default=256,help='batch size for training (default: 256)')    
# -------- ood param. --------
parser.add_argument('--score', choices=['MSP', 'ODIN', 'Energy', 'Mahalanobis', 'GradNorm','RankFeat','React'], default='MSP')
parser.add_argument('--in_data', choices=['CIFAR10', 'ImageNet'], default='CIFAR10')
parser.add_argument('--in_datadir', type=str, help='in data dir')
parser.add_argument('--out_data', choices=['SVHN','LSUN','iSUN','Texture','places365','iNaturalist','SUN','Places'], default='SVHN')
parser.add_argument('--out_datadir', type=str, help='out data dir')
# --------
parser.add_argument('--temperature_energy', default=1, type=int, help='temperature scaling for energy')
parser.add_argument('--temperature_odin', default=1000, type=int, help='temperature scaling for odin')
parser.add_argument('--epsilon_odin', default=0.0014, type=float, help='perturbation magnitude for odin')
parser.add_argument('--mahalanobis_param_path', default='utils_mahalanobis/logs/', help='path to tuned mahalanobis parameters')
parser.add_argument('--temperature_rankfeat', default=1, type=float, help='temperature scaling for RankFeat')
parser.add_argument('--temperature_gradnorm', default=1, type=float, help='temperature scaling for GradNorm')
args = parser.parse_args()

# ======== log writer init. ========
args.dataset = args.in_data
hyperparam = ''
for PATH in args.model_path:
    PATH_name =os.path.split(os.path.split(PATH)[-2])[-1]
    hyperparam = hyperparam + re.findall(r"\d+",PATH_name)[0] + '-'
if not os.path.exists(os.path.join(args.logs_dir,args.dataset,args.arch,'eval-ensemble')):
    os.makedirs(os.path.join(args.logs_dir,args.dataset,args.arch,'eval-ensemble'))
args.logs_path = os.path.join(args.logs_dir,args.dataset,args.arch,'eval-ensemble','%s-'%args.score+hyperparam+'ood.log')
sys.stdout = Logger(filename=args.logs_path,stream=sys.stdout)

# -------- main function
def main():

    # ======== fix random seed ========
    setup_seed(args.seed)

    if args.score == 'GradNorm':
        args.batch_size = 1
    
    # ======== get data set =============
    in_set, out_set, in_loader, out_loader = make_id_ood(args)
    print('-------- DATA INFOMATION --------')
    print('---- in-data : '+args.in_data)
    print('---- out-data: '+args.out_data)

    # ======== load network ========
    net_ensemble = []
    for PATH in args.model_path:
        checkpoint = torch.load(PATH, map_location=torch.device("cpu"))
        net = get_model(args).cuda()
        net.load_state_dict(checkpoint['state_dict'])
        net_ensemble.append(net)
    print('-------- MODEL INFORMATION --------')
    print('---- arch.: '+args.arch)
    print('---- saved path   : ')
    for PATH in args.model_path:
        print('----     '+PATH)
    print('---- inf. seed.: '+str(args.seed))

    # ======== evaluation on Ood ========
    print('Running %s...'%args.score)
    start_time = time.time()
    val_ensemble(net_ensemble, in_loader, out_loader, args)
    duration = time.time() - start_time


    print("Finished. Total running time: {}".format(duration))
    print()



    return

def val_ensemble(net_ensemble, in_loader, out_loader, args):
    
    for net in net_ensemble:
        net.eval()

    if args.score == 'MSP':
        print("Processing in-distribution data...")
        in_scores = iterate_data_msp_ensemble(in_loader, net_ensemble)
        print("Processing out-of-distribution data...")
        out_scores = iterate_data_msp_ensemble(out_loader, net_ensemble)
    elif args.score == 'Energy':
        print("Processing in-distribution data...")
        in_scores = iterate_data_energy_ensemble(in_loader, net_ensemble, args.temperature_energy)
        print("Processing out-of-distribution data...")
        out_scores = iterate_data_energy_ensemble(out_loader, net_ensemble, args.temperature_energy)
    elif args.score == 'ODIN':
        if args.in_data == 'ImageNet':
            args.epsilon_odin = 0.0
        print("Processing in-distribution data...")
        in_scores = iterate_data_odin_ensemble(in_loader, net_ensemble, args.epsilon_odin, args.temperature_odin)
        print("Processing out-of-distribution data...")
        out_scores = iterate_data_odin_ensemble(out_loader, net_ensemble, args.epsilon_odin, args.temperature_odin)
    elif args.score == 'RankFeat':
        print("Processing in-distribution data...")
        in_scores = iterate_data_rankfeat_ensemble(in_loader, net_ensemble, args)
        print("Processing out-of-distribution data...")
        out_scores = iterate_data_rankfeat_ensemble(out_loader, net_ensemble, args)
    elif args.score == 'GradNorm':
        print("Processing in-distribution data...")
        in_scores = iterate_data_gradnorm_ensemble(in_loader, net_ensemble, args)
        print("Processing out-of-distribution data...")
        out_scores = iterate_data_gradnorm_ensemble(out_loader, net_ensemble, args)
    elif args.score == 'Mahalanobis':
        hyperparam = ''
        for idx, PATH in enumerate(args.model_path):
            PATH_name =os.path.split(os.path.split(PATH)[-2])[-1]
            if idx == (len(args.model_path)-1):
                hyperparam = hyperparam + re.findall(r"\d+",PATH_name)[0]
            else:
                hyperparam = hyperparam + re.findall(r"\d+",PATH_name)[0] + '-'
        sample_mean, precision, lr_weights, lr_bias, magnitude = np.load(
            os.path.join(args.mahalanobis_param_path, args.in_data, args.arch, hyperparam, 'results.npy'), allow_pickle=True)
        sample_mean = [s.cuda() for s in sample_mean]
        precision = [p.cuda() for p in precision]

        regressor = LogisticRegressionCV(cv=2).fit([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], [0, 0, 1, 1])

        regressor.coef_ = lr_weights
        regressor.intercept_ = lr_bias

        # print("lr_weights.shape: ", lr_weights.shape)
        # print("lr_bias.shape: ", lr_bias.shape)

        if args.in_data == 'CIFAR10':
            temp_x = torch.rand(2, 3, 32, 32)
            temp_x = Variable(temp_x).cuda()
            temp_list = net_ensemble[0].feature_list(temp_x)[1]
        elif args.in_data == 'ImageNet':
            temp_x = torch.rand(2, 3, 224, 224)
            temp_x = Variable(temp_x).cuda()
            temp_list = net_ensemble[0].module.feature_list(temp_x)[1]
        num_output = len(temp_list)

        print("Processing in-distribution data...")
        in_scores = iterate_data_mahalanobis_ensemble(in_loader, net_ensemble, args.num_classes, sample_mean, precision,
                                             num_output, magnitude, regressor)
        print("Processing out-of-distribution data...")
        out_scores = iterate_data_mahalanobis_ensemble(out_loader, net_ensemble, args.num_classes, sample_mean, precision,
                                              num_output, magnitude, regressor)
    
    in_examples = in_scores.reshape((-1,1))
    out_examples = out_scores.reshape((-1,1))

    auroc, aupr_in, aupr_out, fpr95 = get_measures(in_examples, out_examples)

    print('============Results for {}============'.format(args.score))
    print('AUROC: {}'.format(auroc))
    print('AUPR (In): {}'.format(aupr_in))
    print('AUPR (Out): {}'.format(aupr_out))
    print('FPR95: {}'.format(fpr95))
    
    return


# ======== startpoint
if __name__ == '__main__':
    main()