# ood-mode-ensemble
Mode ensemble for OoD detection. **UPDATING...**


## Mode ensemble for OoD detection in a nutshell

## Overview of the Repo

A description on the files contained in this repo.

### Training
1. `train_c10.py`: training isolated modes w.r.t. different random seeds on CIFAR10
2. `train_imgnet.py`: training isolated modes w.r.t. different random seeds on ImageNet

### Evaluation
1. `eval_clean.py` and `eval_clean_ensemble.py`: evaluation the clean accracy of single modes and ensembling modes, respectively
2. `eval_ood.py` and `eval_ood_ensemble.py`: evaluation the OoD detection performance of single modes and ensembling modes, respectively

### Others
1. `utils_ood.py`: A collection on the utility functions of OoD detectors
2. `utils.py`: Utility functions
3. `utils_knn/`: Utility functions on the kNN method
4. `utils_mahalanobis/`: Utility functions on the Mahalanobis method

## Getting started
Install dependencies
```
conda create -n ood python=3.8
conda activate ood
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch # for Linux
pip install pandas, scipy, scikit-learn, tensorboard
pip install statsmodels
```
Install [faiss](https://github.com/facebookresearch/faiss/tree/main) package following its [docs](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md).

Modify the in-distribution and out-distribution data directories in `./utils_ood.py` as yours.

A full collection of all the training and evaluation commands can be found in [EXPERIMENTS.md](./EXPERIMENTS.md).

## Released trained-models

Our models trained w.r.t. different random seeds, including R18-C10, WRN28X10-C10, R50-ImgNet, DN121-ImgNet and T2T-ViT-14-ImgNet are released [here](https://drive.google.com/drive/folders/123fa0dEG-t0qyLjIEgevCyoSvGFQ0iyt?usp=sharing).

Download these models and put them in `./save/` as follows
```
ood-mode-ensemble
├── model
├── utils_knn
├── utils_mahalanobis
├── save
|   ├── CIFAR10
|   └── ImageNet
|       ├── DN121
|       |   ├── seed-1000
|       |   |   └──checkpoint.pth.tar
|       |   ├── seed-2000
|       |   ├── ...
|       |   └── seed-5000
|       ├── R50
|       └── t2tvit 
├── ...
```

## References
The loss landscape visualization techniques follow [mode-connectivity](https://github.com/timgaripov/dnn-mode-connectivity) and [loss-surface](https://github.com/tomgoldstein/loss-landscape).

## 

If u have problems about the code or paper, u could contact me (fanghenshao@sjtu.edu.cn) or raise issues here.

If u find the code useful, welcome to fork and star ⭐ this repo and cite our paper! :)