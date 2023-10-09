import sys
sys.path.append("..")
sys.path.append("../model")

import argparse
import torch
import time
import torchvision as tv
import numpy as np
import os
import re
from torch.autograd import Variable
# from mahalanobis_lib import sample_estimator, get_Mahalanobis_score
from mahalanobis_lib import sample_estimator_ensemble, get_Mahalanobis_score_ensemble
import torch.nn as nn
from sklearn.linear_model import LogisticRegressionCV
import torchvision.models as models

from utils_ood import get_measures
from utils import get_model, get_datasets

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)



def tune_mahalanobis_hyperparams_ensemble(args, model_ensemble, num_classes, train_loader, val_loader):

    hyperparam = ''
    for idx, PATH in enumerate(args.model_path):
        PATH_name =os.path.split(os.path.split(PATH)[-2])[-1]
        if idx == (len(args.model_path)-1):
            hyperparam = hyperparam + re.findall(r"\d+",PATH_name)[0]
        else:
            hyperparam = hyperparam + re.findall(r"\d+",PATH_name)[0] + '-'
    save_dir = os.path.join(args.logs_dir, args.in_data, args.arch, hyperparam)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for model in model_ensemble:
        model.eval()

    # set information about feature extaction
    if args.in_data == 'CIFAR10':
        temp_x = torch.randn(2, 3, 32, 32)
    elif args.in_data == 'ImageNet':
        temp_x = torch.rand(2, 3, 224, 224)
    temp_x = Variable(temp_x).cuda()
    temp_list = model.feature_list(temp_x)[1]
    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1

    print('get sample mean and covariance')
    filename = os.path.join(save_dir, 'mean_and_precision.npy')

    FORCE_RUN = True
    if FORCE_RUN or not os.path.exists(filename):
        sample_mean, precision = sample_estimator_ensemble(model_ensemble, num_classes, feature_list, train_loader)
        np.save(filename, np.array([sample_mean, precision]))
        print("save.")

    sample_mean, precision = np.load(filename, allow_pickle=True)
    sample_mean = [s.cuda() for s in sample_mean]
    precision = [p.cuda() for p in precision]

    print('train logistic regression model')
    m = 500

    train_in = []
    train_in_label = []
    train_out = []

    val_in = []
    val_in_label = []
    val_out = []

    cnt = 0
    for data, target in val_loader:
        data = data.numpy()
        target = target.numpy()
        for x, y in zip(data, target):
            cnt += 1
            if cnt <= m:
                train_in.append(x)
                train_in_label.append(y)
            elif cnt <= 2*m:
                val_in.append(x)
                val_in_label.append(y)

            if cnt == 2*m:
                break
        if cnt == 2*m:
            break

    print('In {} {}'.format(len(train_in), len(val_in)))

    criterion = nn.CrossEntropyLoss().cuda()
    # adv_noise = 0.05
    if args.in_data == 'CIFAR10':
        adv_noise = 0.05
    elif args.in_data == 'ImageNet':
        adv_noise = 0.001

    # args.batch_size = args.batch
    for i in range(int(m/args.batch_size) + 1):
        if i*args.batch_size >= m:
            break
        data = torch.tensor(train_in[i*args.batch_size:min((i+1)*args.batch_size, m)])
        target = torch.tensor(train_in_label[i*args.batch_size:min((i+1)*args.batch_size, m)])
        data = data.cuda()
        target = target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        # output = model(data)

        # model.zero_grad()
        # inputs = Variable(data.data, requires_grad=True).cuda()
        # output = model(inputs)
        # loss = criterion(output, target)
        # loss.backward()
        inputs = Variable(data.data, requires_grad=True).cuda()
        loss = 0
        for model in model_ensemble:
            model.zero_grad()
            loss = loss + criterion(model(inputs), target)
        loss = loss / len(model_ensemble)
        loss.backward()


        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float()-0.5)*2

        adv_data = torch.add(input=inputs.data, other=gradient, alpha=adv_noise)
        adv_data = torch.clamp(adv_data, 0.0, 1.0)

        train_out.extend(adv_data.cpu().numpy())

    for i in range(int(m/args.batch_size) + 1):
        if i*args.batch_size >= m:
            break
        data = torch.tensor(val_in[i*args.batch_size:min((i+1)*args.batch_size, m)])
        target = torch.tensor(val_in_label[i*args.batch_size:min((i+1)*args.batch_size, m)])
        data = data.cuda()
        target = target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        # output = model(data)

        # model.zero_grad()
        # inputs = Variable(data.data, requires_grad=True).cuda()
        # output = model(inputs)
        # loss = criterion(output, target)
        # loss.backward()
        inputs = Variable(data.data, requires_grad=True).cuda()
        loss = 0
        for model in model_ensemble:
            model.zero_grad()
            loss = loss + criterion(model(inputs), target)
        loss = loss / len(model_ensemble)
        loss.backward()

        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float()-0.5)*2

        adv_data = torch.add(input=inputs.data, other=gradient, alpha=adv_noise)
        adv_data = torch.clamp(adv_data, 0.0, 1.0)

        val_out.extend(adv_data.cpu().numpy())

    print('Out {} {}'.format(len(train_out),len(val_out)))

    train_lr_data = []
    train_lr_label = []
    train_lr_data.extend(train_in)
    train_lr_label.extend(np.zeros(m))
    train_lr_data.extend(train_out)
    train_lr_label.extend(np.ones(m))
    train_lr_data = torch.tensor(train_lr_data)
    train_lr_label = torch.tensor(train_lr_label)

    best_fpr = 1.1
    best_magnitude = 0.0

    for magnitude in [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]:
        train_lr_Mahalanobis = []
        total = 0
        for data_index in range(int(np.floor(train_lr_data.size(0) / args.batch_size)) + 1):
            if total >= 2*m:
                break
            data = train_lr_data[total : total + args.batch_size].cuda()
            total += args.batch_size
            Mahalanobis_scores = get_Mahalanobis_score_ensemble(data, model_ensemble, num_classes, sample_mean, precision, num_output, magnitude)
            train_lr_Mahalanobis.extend(Mahalanobis_scores)

        train_lr_Mahalanobis = np.asarray(train_lr_Mahalanobis, dtype=np.float32)
        regressor = LogisticRegressionCV(n_jobs=-1).fit(train_lr_Mahalanobis, train_lr_label)

        print('Logistic Regressor params: {} {}'.format(regressor.coef_, regressor.intercept_))

        t0 = time.time()
        f1 = open(os.path.join(save_dir, "confidence_mahalanobis_In.txt"), 'w')
        f2 = open(os.path.join(save_dir, "confidence_mahalanobis_Out.txt"), 'w')

        ########################################In-distribution###########################################
        print("Processing in-distribution images")

        count = 0
        all_confidence_scores_in, all_confidence_scores_out = [], []
        for i in range(int(m/args.batch_size) + 1):
            if i * args.batch_size >= m:
                break
            images = torch.tensor(val_in[i * args.batch_size : min((i+1) * args.batch_size, m)]).cuda()
            # if j<1000: continue
            batch_size = images.shape[0]
            Mahalanobis_scores = get_Mahalanobis_score_ensemble(images, model_ensemble, num_classes, sample_mean, precision, num_output, magnitude)
            confidence_scores_in = -regressor.predict_proba(Mahalanobis_scores)[:, 1]
            all_confidence_scores_in.extend(confidence_scores_in)

            for k in range(batch_size):
                f1.write("{}\n".format(confidence_scores_in[k]))

            count += batch_size
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, m, time.time()-t0))
            t0 = time.time()

        ###################################Out-of-Distributions#####################################
        t0 = time.time()
        print("Processing out-of-distribution images")
        count = 0

        for i in range(int(m/args.batch_size) + 1):
            if i * args.batch_size >= m:
                break
            images = torch.tensor(val_out[i * args.batch_size : min((i+1) * args.batch_size, m)]).cuda()
            # if j<1000: continue
            batch_size = images.shape[0]

            Mahalanobis_scores = get_Mahalanobis_score_ensemble(images, model_ensemble, num_classes, sample_mean, precision, num_output, magnitude)

            confidence_scores_out = -regressor.predict_proba(Mahalanobis_scores)[:, 1]
            all_confidence_scores_out.extend(confidence_scores_out)

            for k in range(batch_size):
                f2.write("{}\n".format(confidence_scores_out[k]))

            count += batch_size
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, m, time.time()-t0))
            t0 = time.time()

        f1.close()
        f2.close()

        # results = metric(save_dir, stypes)
        # print_results(results, stypes)
        # fpr = results['mahalanobis']['FPR']
        all_confidence_scores_in = np.array(all_confidence_scores_in).reshape(-1, 1)
        all_confidence_scores_out = np.array(all_confidence_scores_out).reshape(-1, 1)
        print("all_confidence_scores_in.shape: ", all_confidence_scores_in.shape)
        print("all_confidence_scores_out.shape: ", all_confidence_scores_out.shape)

        _, _, _, fpr = get_measures(all_confidence_scores_in, all_confidence_scores_out)

        if fpr < best_fpr:
            best_fpr = fpr
            best_magnitude = magnitude
            best_regressor = regressor

    print('Best Logistic Regressor params: {} {}'.format(best_regressor.coef_, best_regressor.intercept_))
    print('Best magnitude: {}'.format(best_magnitude))

    return sample_mean, precision, best_regressor, best_magnitude


def main(args):
    # logger = log.setup_logger(args)

    # Lets cuDNN benchmark conv implementations and choose the fastest.
    # Only good if sizes stay the same within the main loop!
    torch.backends.cudnn.benchmark = True

    # train_set, val_set, train_loader, val_loader = mktrainval(args)

    train_loader, val_loader = get_datasets(args)

    # print(f"Loading model from {args.model_path}")
    #model = resnetv2.KNOWN_MODELS[args.model](head_size=len(train_set.classes))
    #state_dict = torch.load(args.model_path)
    #model.load_state_dict_custom(state_dict['model'])
    #model = squeezenet1_0(pretrained=True)
    # model = t2t_vit_24()
    # state_dict = torch.load('82.3_T2T_ViT_24.pth.tar')
    # model.load_state_dict(state_dict['state_dict_ema'])
    net_ensemble = []
    for PATH in args.model_path:
        print(f"Loading model from {PATH}")
        checkpoint = torch.load(PATH, map_location=torch.device("cpu"))
        net = get_model(args).cuda()
        net.load_state_dict(checkpoint['state_dict'])
        net_ensemble.append(net)

    # logger.info("Moving model onto all GPUs")
    #model = torch.nn.DataParallel(model)
    # model = model.cuda()

    print('Tuning hyper-parameters...')
    sample_mean, precision, best_regressor, best_magnitude \
        = tune_mahalanobis_hyperparams_ensemble(args, net_ensemble, len(val_loader.dataset.classes), train_loader, val_loader)

    print('saving results...')
    # save_dir = os.path.join(args.logs_dir, args.in_data)
    hyperparam = ''
    for idx, PATH in enumerate(args.model_path):
        PATH_name =os.path.split(os.path.split(PATH)[-2])[-1]
        if idx == (len(args.model_path)-1):
            hyperparam = hyperparam + re.findall(r"\d+",PATH_name)[0]
        else:
            hyperparam = hyperparam + re.findall(r"\d+",PATH_name)[0] + '-'
    save_dir = os.path.join(args.logs_dir, args.in_data, args.arch, hyperparam)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    sample_mean = [s.cpu() for s in sample_mean]
    precision = [p.cpu() for p in precision]
    np.save(os.path.join(save_dir, 'results'),
            np.array([sample_mean, precision, best_regressor.coef_, best_regressor.intercept_, best_magnitude]))


if __name__ == "__main__":
    # parser = arg_parser() 
    parser = argparse.ArgumentParser(description='Tuning Mahalanobis')
    parser.add_argument('--dataset',type=str,default='CIFAR10',help='data set name')
    parser.add_argument('--data_dir',type=str,default='./data/CIFAR10/',help='data directory')
    parser.add_argument('--batch_size',type=int,default=256,help='batch size for training (default: 256)')  
    
    parser.add_argument('--arch',type=str,default='vgg16',help='model architecture')
    parser.add_argument('--model_path',type=str,help='saved model path',nargs='+')

    parser.add_argument('--in_data', choices=['CIFAR10', 'ImageNet'], default='CIFAR10')
    parser.add_argument('--logs_dir',type=str,default='./logs/',help='logs directory')

    main(parser.parse_args())
