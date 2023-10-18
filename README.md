# ood-mode-ensemble
Mode ensemble for OoD detection. **UPDATING**


## Mode ensemble for OoD detection in a nutshell

## Overview of the Repo

A description on the files contained in this repo.

### Training
1. `train_c10.py`: training isolated modes w.r.t. different random seeds on CIFAR10

### Evaluation
1. `eval_clean.py` and `eval_clean_ensemble.py`: evaluation the clean accracy of single modes and independent modes
2. `eval_ood.py` and `eval_ood_ensemble.py`: evaluation the OoD detection performance of single modes and independent modes

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

Modify the in-distribution and out-distribution data directory in `utils_ood.py` as yours.

## Examples

## Released trained-models

## References
The loss landscape visualization techniques follow [mode-connectivity](https://github.com/timgaripov/dnn-mode-connectivity) and [loss-surface](https://github.com/tomgoldstein/loss-landscape).