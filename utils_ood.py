import torch
from torch.autograd import Variable

import torchvision as tv
import torchvision.transforms as transforms

import numpy as np
import sklearn.metrics as sk

from utils_mahalanobis.mahalanobis_lib import get_Mahalanobis_score
from utils_mahalanobis.mahalanobis_lib import get_Mahalanobis_score_ensemble

def make_id_ood(args):
    """Returns train and validation datasets."""
    if args.in_data == 'CIFAR10':
        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        args.in_datadir = '~/fangkun/data/CIFAR10'
        in_set = tv.datasets.CIFAR10(args.in_datadir, train=False, transform=test_transform, download=True)

        if args.out_data == 'SVHN':
            args.out_datadir = '~/fangkun/data/ood_data/svhn/'
            out_set = tv.datasets.SVHN(args.out_datadir, split='test', transform=test_transform, download=False)
        elif args.out_data == 'LSUN' or args.out_data == 'iSUN' or args.out_data == 'places365':
            args.out_datadir = '~/fangkun/data/ood_data/{}'.format(args.out_data)
            out_set = tv.datasets.ImageFolder(args.out_datadir, transform=test_transform)
        elif args.out_data == 'Texture':
            args.out_datadir = '~/fangkun/data/ood_data/dtd/images'
            out_set = tv.datasets.ImageFolder(args.out_datadir, transform=test_transform)


    elif args.in_data == 'ImageNet':
        args.in_datadir = '~/imagenet/ILSVRC2012_img_val'
        if args.out_data == 'iNaturalist' or args.out_data == 'SUN' or args.out_data == 'Places':
            args.out_datadir = "~/fangkun/data/ood_data/{}".format(args.out_data)
        elif args.out_data == 'Texture':
            args.out_datadir = '~/fangkun/data/ood_data/dtd/images'

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])

        in_set = tv.datasets.ImageFolder(args.in_datadir, test_transform)
        out_set = tv.datasets.ImageFolder(args.out_datadir, test_transform)

    print(f"Using an in-distribution set {args.in_data} with {len(in_set)} images.")
    print(f"Using an out-of-distribution set {args.out_data} with {len(out_set)} images.")

    in_loader = torch.utils.data.DataLoader(
        in_set, batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=True, drop_last=False)

    out_loader = torch.utils.data.DataLoader(
        out_set, batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=True, drop_last=False)
    
    args.num_classes = len(in_set.classes)

    return in_set, out_set, in_loader, out_loader

def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level, pos_label=1.):
    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])



def get_measures(in_examples, out_examples):
    num_in = in_examples.shape[0]
    num_out = out_examples.shape[0]

    # logger.info("# in example is: {}".format(num_in))
    # logger.info("# out example is: {}".format(num_out))

    labels = np.zeros(num_in + num_out, dtype=np.int32)
    labels[:num_in] += 1

    examples = np.squeeze(np.vstack((in_examples, out_examples)))
    aupr_in = sk.average_precision_score(labels, examples)
    auroc = sk.roc_auc_score(labels, examples)

    recall_level = 0.95
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    labels_rev = np.zeros(num_in + num_out, dtype=np.int32)
    labels_rev[num_in:] += 1
    examples = np.squeeze(-np.vstack((in_examples, out_examples)))
    aupr_out = sk.average_precision_score(labels_rev, examples)
    return auroc, aupr_in, aupr_out, fpr

# ======== Maximum Logit Score
def iterate_data_msp(data_loader, model):
    confs = []
    m = torch.nn.Softmax(dim=-1).cuda()
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            logits = model(x)
            conf, _ = torch.max(m(logits), dim=-1)
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs)

def iterate_data_msp_ensemble(data_loader, model_ensemble):
    confs = []
    m = torch.nn.Softmax(dim=-1).cuda()
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            logits = 0
            for model in model_ensemble:
                logits = logits + model(x)
            logits = logits / len(model_ensemble)
            conf, _ = torch.max(m(logits), dim=-1)
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs)

# ======== Energy Score
def iterate_data_energy(data_loader, model, temper):
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            # compute output, measure accuracy and record loss.
            logits = model(x)

            conf = temper * torch.logsumexp(logits / temper, dim=1)
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs)

def iterate_data_energy_ensemble(data_loader, model_ensemble, temper):
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            # compute output, measure accuracy and record loss.
            logits = 0
            for model in model_ensemble:
                logits = logits + model(x)
            logits = logits / len(model_ensemble)

            conf = temper * torch.logsumexp(logits / temper, dim=1)
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs)

# ======== ODIN Score
def iterate_data_odin(data_loader, model, epsilon, temper): #, is_ImgNet):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    confs = []
    for b, (x, y) in enumerate(data_loader):
        x = Variable(x.cuda(), requires_grad=True)
        outputs = model(x)

        # if the ind data is imagenet, we dot not perturb the input
        # if not is_ImgNet:
        maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)
        outputs = outputs / temper

        labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
        loss = criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(x.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        # Adding small perturbations to images
        tempInputs = torch.add(x.data, -epsilon, gradient)
        outputs = model(Variable(tempInputs))
        outputs = outputs / temper
        
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

        confs.extend(np.max(nnOutputs, axis=1))
    return np.array(confs)

def iterate_data_odin_ensemble(data_loader, model_ensemble, epsilon, temper):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    confs = []
    for b, (x, y) in enumerate(data_loader):
        x = Variable(x.cuda(), requires_grad=True)

        avg_logits = 0
        for model in model_ensemble:
            outputs = model(x)

            maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)
            outputs = outputs / temper

            labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
            loss = criterion(outputs, labels)
            loss.backward()

            # Normalizing the gradient to binary in {0, 1}
            gradient = torch.ge(x.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2

            # Adding small perturbations to images
            tempInputs = torch.add(x.data, -epsilon, gradient)
            outputs = model(Variable(tempInputs))

            avg_logits = avg_logits + outputs
        
        avg_logits = avg_logits / len(model_ensemble)
        outputs = avg_logits

        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

        confs.extend(np.max(nnOutputs, axis=1))
    return np.array(confs)

# ======== Mahalanobis Distance Score
def iterate_data_mahalanobis(data_loader, model, num_classes, sample_mean, precision,
                             num_output, magnitude, regressor):
    confs = []
    for b, (x, y) in enumerate(data_loader):
        # if b % 10 == 0:
        #     logger.info('{} batches processed'.format(b))
        x = x.cuda()

        Mahalanobis_scores = get_Mahalanobis_score(x, model, num_classes, sample_mean, precision, num_output, magnitude)
        # print("Mahal_scores.shape: ", Mahalanobis_scores.shape)
        # print("num_output: ", num_output)
        # print("regressor.n_features: ", regressor.n_features_in_)
        # print("regressor.coef_", regressor.coef_.shape)
        scores = -regressor.predict_proba(Mahalanobis_scores)[:, 1]
        confs.extend(scores)
    return np.array(confs)

def iterate_data_mahalanobis_ensemble(data_loader, model_ensemble, num_classes, sample_mean, precision,
                             num_output, magnitude, regressor):
    confs = []
    for b, (x, y) in enumerate(data_loader):
        # if b % 10 == 0:
        #     logger.info('{} batches processed'.format(b))
        x = x.cuda()

        Mahalanobis_scores = get_Mahalanobis_score_ensemble(x, model_ensemble, num_classes, sample_mean, precision, num_output, magnitude)
        # print("Mahal_scores.shape: ", Mahalanobis_scores.shape)
        # print("num_output: ", num_output)
        # print("regressor.n_features: ", regressor.n_features_in_)
        # print("regressor.coef_", regressor.coef_.shape)
        scores = -regressor.predict_proba(Mahalanobis_scores)[:, 1]
        confs.extend(scores)
    return np.array(confs)

# ======== RankFeat Score
def iterate_data_rankfeat(data_loader, model, args):
    confs = []
    for b, (x, y) in enumerate(data_loader):
        # if b % 100 == 0:
            # print('{} batches processed'.format(b))
        inputs = x.cuda()

        #Logit of Block 4 feature
        # feat1 = model.intermediate_forward(inputs,layer_index=4)
        feat1 = model.module.intermediate_forward(inputs,layer_index=4)
        B, C, H, W = feat1.size()
        feat1 = feat1.view(B, C, H * W)
        u,s,v = torch.linalg.svd(feat1,full_matrices=False)
        feat1 = feat1 - s[:,0:1].unsqueeze(2)*u[:,:,0:1].bmm(v[:,0:1,:])
        #if you want to use PI for acceleration, comment the above 2 lines and uncomment the line below
        #feat1 = feat1 - power_iteration(feat1, iter=20)
        feat1 = feat1.view(B,C,H,W)
        # logits1 = model.forward_head(feat1)
        logits1 = model.module.forward_head(feat1)

        # Logit of Block 3 feature
        # feat2 = model.intermediate_forward(inputs, layer_index=3)
        feat2 = model.module.intermediate_forward(inputs, layer_index=3)
        B, C, H, W = feat2.size()
        feat2 = feat2.view(B, C, H * W)
        u, s, v = torch.linalg.svd(feat2,full_matrices=False)
        feat2 = feat2 - s[:, 0:1].unsqueeze(2) * u[:, :, 0:1].bmm(v[:, 0:1, :])
        #if you want to use PI for acceleration, comment the above 2 lines and uncomment the line below
        #feat2 = feat2 - power_iteration(feat2, iter=20)
        feat2 = feat2.view(B, C, H, W)
        if args.in_data == "ImageNet" and args.arch == 'R50':
            feat2 = model.module.layer4(feat2)
            logits2 = model.module.forward_head(feat2)
        elif args.in_data == "ImageNet" and args.arch == 'DN121':
            feat2 = model.module.features.denseblock4(feat2)
            feat2 = model.module.features.norm5(feat2)
            logits2 = model.module.forward_head(feat2)

        #Fusion at the logit space
        logits = (logits1+logits2) / 2
        conf = args.temperature_rankfeat * torch.logsumexp(logits / args.temperature_rankfeat, dim=1)
        confs.extend(conf.data.cpu().numpy())

    return np.array(confs)

def iterate_data_rankfeat_ensemble(data_loader, model_ensemble, args):
    confs = []
    for b, (x, y) in enumerate(data_loader):
        # if b % 100 == 0:
            # print('{} batches processed'.format(b))
        inputs = x.cuda()

        avg_logits = 0
        for model in model_ensemble:

            #Logit of Block 4 feature
            # feat1 = model.intermediate_forward(inputs,layer_index=4)
            feat1 = model.module.intermediate_forward(inputs,layer_index=4)
            B, C, H, W = feat1.size()
            feat1 = feat1.view(B, C, H * W)
            u,s,v = torch.linalg.svd(feat1,full_matrices=False)
            feat1 = feat1 - s[:,0:1].unsqueeze(2)*u[:,:,0:1].bmm(v[:,0:1,:])
            #if you want to use PI for acceleration, comment the above 2 lines and uncomment the line below
            #feat1 = feat1 - power_iteration(feat1, iter=20)
            feat1 = feat1.view(B,C,H,W)
            # logits1 = model.forward_head(feat1)
            logits1 = model.module.forward_head(feat1)

            # Logit of Block 3 feature
            # feat2 = model.intermediate_forward(inputs, layer_index=3)
            feat2 = model.module.intermediate_forward(inputs, layer_index=3)
            B, C, H, W = feat2.size()
            feat2 = feat2.view(B, C, H * W)
            u, s, v = torch.linalg.svd(feat2,full_matrices=False)
            feat2 = feat2 - s[:, 0:1].unsqueeze(2) * u[:, :, 0:1].bmm(v[:, 0:1, :])
            #if you want to use PI for acceleration, comment the above 2 lines and uncomment the line below
            #feat2 = feat2 - power_iteration(feat2, iter=20)
            feat2 = feat2.view(B, C, H, W)
            if args.in_data == "ImageNet" and args.arch == 'R50':
                feat2 = model.module.layer4(feat2)
                logits2 = model.module.forward_head(feat2)
            elif args.in_data == "ImageNet" and args.arch == 'DN121':
                feat2 = model.module.features.denseblock4(feat2)
                feat2 = model.module.features.norm5(feat2)
                logits2 = model.module.forward_head(feat2)

            #Fusion at the logit space
            logits = (logits1+logits2) / 2

            avg_logits = avg_logits + logits
        avg_logits = avg_logits / len(model_ensemble)

        conf = args.temperature_rankfeat * torch.logsumexp(avg_logits / args.temperature_rankfeat, dim=1)
        confs.extend(conf.data.cpu().numpy())

    return np.array(confs)

# ======== GradNorm Score
def iterate_data_gradnorm(data_loader, model, args): # temperature, num_classes):
    confs = []
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()
    for b, (x, y) in enumerate(data_loader):
        # if b % 100 == 0:
            # print('{} batches processed'.format(b))
        inputs = Variable(x.cuda(), requires_grad=True)

        model.zero_grad()
        outputs = model(inputs)
        targets = torch.ones((inputs.shape[0], args.num_classes)).cuda()
        outputs = outputs / args.temperature_gradnorm
        loss = torch.mean(torch.sum(-targets * logsoftmax(outputs), dim=-1))

        loss.backward()

        if args.arch == 'R50':
            layer_grad = model.module.fc.weight.grad.data
        elif args.arch == 'DN121':
            layer_grad = model.module.classifier.weight.grad.data
        # layer_grad = model.linear.weight.grad.data

        layer_grad_norm = torch.sum(torch.abs(layer_grad)).cpu().numpy()
        confs.append(layer_grad_norm)

    return np.array(confs)

def iterate_data_gradnorm_ensemble(data_loader, model_ensemble, args): # temperature, num_classes):
    confs = []
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()
    for b, (x, y) in enumerate(data_loader):
        # if b % 100 == 0:
            # print('{} batches processed'.format(b))
        inputs = Variable(x.cuda(), requires_grad=True)

        outputs = 0
        for model in model_ensemble:
            model.zero_grad()
            outputs = outputs + model(inputs)
        outputs = outputs / len(model_ensemble) / args.temperature_gradnorm
        targets = torch.ones((inputs.shape[0], args.num_classes)).cuda()
        loss = torch.mean(torch.sum(-targets * logsoftmax(outputs), dim=-1))
        loss.backward()

        # layer_grad = model.module.classifier.weight.grad.data

        avg_layer_grad_norm = 0
        if args.in_data == 'ImageNet' and args.arch == 'R50':
            for model in model_ensemble:
                layer_grad = model.module.fc.weight.grad.data
                avg_layer_grad_norm = avg_layer_grad_norm + torch.sum(torch.abs(layer_grad)).cpu().numpy()
        if args.in_data == 'ImageNet' and args.arch == 'DN121':
            for model in model_ensemble:
                layer_grad = model.module.classifier.weight.grad.data
                avg_layer_grad_norm = avg_layer_grad_norm + torch.sum(torch.abs(layer_grad)).cpu().numpy()
        confs.append(avg_layer_grad_norm/len(model_ensemble))

    return np.array(confs)