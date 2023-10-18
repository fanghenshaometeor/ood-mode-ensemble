import torch
from torch.autograd import Variable

import torchvision as tv
import torchvision.transforms as transforms


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
        in_set_train = tv.datasets.CIFAR10(args.in_datadir, train=True, transform=test_transform, download=True)

        if args.out_data == 'SVHN':
            args.out_datadir = '~/fangkun/data/ood_data/svhn/'
            out_set = tv.datasets.SVHN(args.out_datadir, split='test', transform=test_transform, download=False)
        elif args.out_data in ['LSUN','iSUN','places365','LSUN_FIX','ImageNet_FIX','ImageNet_resize']:
            args.out_datadir = '~/fangkun/data/ood_data/{}'.format(args.out_data)
            out_set = tv.datasets.ImageFolder(args.out_datadir, transform=test_transform)
        elif args.out_data == 'Texture':
            args.out_datadir = '~/fangkun/data/ood_data/dtd/images'
            out_set = tv.datasets.ImageFolder(args.out_datadir, transform=test_transform)
        elif args.out_data == 'CIFAR100':
            args.out_datadir = '~/fangkun/data/CIFAR100'
            out_set = tv.datasets.CIFAR100(args.out_datadir, train=False, download=True, transform=test_transform)


    elif args.in_data == 'ImageNet':
        args.in_datadir = '~/imagenet/ILSVRC2012_img_val'
        args.in_datadir_train = '~/imagenet/ILSVRC2012_img_train'
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
        in_set_train = tv.datasets.ImageFolder(args.in_datadir_train, test_transform)

    print(f"Using an in-distribution set {args.in_data} with {len(in_set)} images.")
    print(f"Using an out-of-distribution set {args.out_data} with {len(out_set)} images.")

    in_loader = torch.utils.data.DataLoader(
        in_set, batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=True, drop_last=False)

    in_loader_train = torch.utils.data.DataLoader(
        in_set_train, batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=True, drop_last=False)

    out_loader = torch.utils.data.DataLoader(
        out_set, batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=True, drop_last=False)
    
    args.num_classes = len(in_set.classes)

    return in_set, in_set_train, out_set, in_loader, in_loader_train, out_loader