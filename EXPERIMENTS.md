# Training and Evaluation commands

All the commands to reproduce the reported results in our paper are listed below.

## Training isolated modes on CIFAR10 and ImageNet

### CIFAR10

To train isolated modes of R18 and WRN28X10 on CIFAR10, run the following BASH command
```
for seed in 1000 1100 1200 1300 1400 2000 2100 2200 2300 2400
do
for arch in R18 WRN28X10
do
CUDA_VISIBLE_DEVICES=0 python train_c10.py \
 --arch ${arch} --dataset CIFAR10 --data_dir YOUR_CIFAR10_DIR \
 --seed ${seed}
done
done
```

### ImageNet

To train isolated modes of R50 and DN121 on ImageNet, run the following BASH command
```
for seed in 1000 2000 3000 4000 5000
do
python train_imgnet.py -a resnet50 \
 --dist-url 'tcp://127.0.0.1:12346' \
 --dist-backend 'nccl' --multiprocessing-distributed \
 --world-size 1 --rank 0 YOUR_IMAGENET_DIR --seed ${seed} --batch-size 1000 --workers 16
done
```
```
for seed in 1000 2000 3000 4000 5000
do
python train_imgnet.py -a densenet121 \
 --dist-url 'tcp://127.0.0.1:12346' \
 --dist-backend 'nccl' --multiprocessing-distributed \
 --world-size 1 --rank 0 YOUR_IMAGENET_DIR --seed ${seed} --batch-size 800 --workers 16
done
```
Our models of R50 and DN121 on ImageNet are trained parallelly on 4 V100 GPUs.

After the training finishes, it is recommended to change the directory name `./save/ImageNet/resnet50/` and `./save/ImageNet/densenet121/` to `./save/ImageNet/R50/` and `./save/ImageNet/DN121/`, respectively.

All the trained models have been released [here](https://drive.google.com/drive/folders/123fa0dEG-t0qyLjIEgevCyoSvGFQ0iyt?usp=sharing).

## Evaluation on the in-distribution test data

To evaluate the test accuracy of independent modes on CIFAR10 and ImageNet, run the following BASH commands

```
for seed in 1000 1100 1200 1300 1400 2000 2100 2200 2300 2400
do
for arch in R18 WRN28X10
do
CUDA_VISIBLE_DEVICES=0 python eval_clean.py \
 --arch ${arch} --dataset CIFAR10 --data_dir YOUR_CIFAR10_DIR \
 --model_path "./save/CIFAR10/${arch}/seed-${seed}/epoch150.pth"
done
done
```

```
for seed in 1000 2000 3000 4000 5000
do
for arch in R50 DN121
do
CUDA_VISIBLE_DEVICES=0 python eval_clean.py \
 --arch ${arch} --dataset ImageNet --data_dir YOUR_IMAGENET_DIR \
 --model_path "./save/ImageNet/${arch}/seed-${seed}/checkpoint.pth.tar"
done
done
```

## Evaluation on out-of-distribution detection

You should prepare all the common OoD data sets and remember to modify the in-distribution and out-distribution data directories in `./utils_ood.py` as yours.

### Detection performance of single modes

To evaluate the OoD detection performance of independent modes, run the following BASH commands

#### MSP, ODIN, Energy

```
for seed in 1000 1100 1200 1300 1400 2000 2100 2200 2300 2400
do
for arch in R18 WRN28X10
do
for score in MSP ODIN Energy
do
for out_data in SVHN LSUN iSUN Texture places365
do
CUDA_VISIBLE_DEVICES=0 python eval_ood.py \
 --arch ${arch} --score ${score} \
 --in_data CIFAR10 --out_data ${out_data} \
 --model_path "./save/CIFAR10/${arch}/seed-${seed}/epoch150.pth"
done
done
done
done
```

```
for seed in 1000 2000 3000 4000 5000
do
for arch in R50 DN121
do
for score in MSP ODIN Energy
do
for out_data in iNaturalist SUN Places Texture
do
CUDA_VISIBLE_DEVICES=1 python eval_ood.py \
 --arch ${arch} --score ${score} \
 --in_data ImageNet --out_data ${out_data} \
 --model_path "./save/ImageNet/${arch}/seed-${seed}/checkpoint.pth.tar" 
done
done
done
done
```

#### GradNorm, RankFeat

```
for seed in 1000 2000 3000 4000 5000
do
for arch in R50 DN121
do
for score in GradNorm RankFeat
do
for out_data in iNaturalist SUN Places Texture
do
CUDA_VISIBLE_DEVICES=0 python eval_ood.py \
 --arch ${arch} --score ${score} \
 --in_data ImageNet --out_data ${out_data} \
 --model_path "./save/ImageNet/${arch}/seed-${seed}/checkpoint.pth.tar" 
done
done
done
done
```

### Detection performance of mode ensemble

We give an example on how to run the `eval_ood_ensemble.py` on ensembling 3 modes.

```
for arch in R18 WRN28X10
do
for out_data in SVHN LSUN iSUN Texture places365
do
CUDA_VISIBLE_DEVICES=0 python eval_ood_ensemble.py \
 --arch ${arch} --score Energy \
 --in_data CIFAR10 --out_data ${out_data} \
 --model_path "./save/CIFAR10/${arch}/seed-1000/epoch150.pth" \
 "./save/CIFAR10/${arch}/seed-1200/epoch150.pth" \
 "./save/CIFAR10/${arch}/seed-2000/epoch150.pth"
done
done
```

```
for arch in R50 DN121
do
for out_data in iNaturalist SUN Places Texture
do
CUDA_VISIBLE_DEVICES=0 python eval_ood_ensemble.py \
 --arch ${arch} --score RankFeat \
 --in_data ImageNet --out_data ${out_data} \
 --model_path "./save/ImageNet/${arch}/seed-1000/checkpoint.pth.tar" \
 "./save/ImageNet/${arch}/seed-3000/checkpoint.pth.tar" \
 "./save/ImageNet/${arch}/seed-5000/checkpoint.pth.tar"
done
done
```

## Evaluation on the kNN detector

Please change the in-distribution and out-distribution data directories in `./utils_knn/utils_data.py` as yours.

### kNN on single modes

To evaluate the kNN detector on single modes, the steps are:
1. Feature extraction
```
cd utils_knn
for seed in 1000 1100 1200 1300 1400 2000 2100 2200 2300 2400
do
for arch in R18 WRN28X10
do
for out_data in SVHN LSUN iSUN Texture places365
do
CUDA_VISIBLE_DEVICES=0 python feat_extract.py \
 --arch ${arch} --in_data CIFAR10 --out_data ${out_data} \
 --model_path "../save/CIFAR10/${arch}/seed-${seed}/epoch150.pth" \
done
done
done
```

2. Perform nearest neighbor search
```
for seed in 1000 1100 1200 1300 1400 2000 2100 2200 2300 2400 
do
for arch in R18 WRN28X10
do
CUDA_VISIBLE_DEVICES=0 python knn.py \
 --arch ${arch} --in_data CIFAR10 --train_seed ${seed} \
 --out_datasets SVHN LSUN iSUN Texture places365
done
done
```

### kNN on mode ensemble

For ensembling multiple modes, the steps of kNN are:
1. Feature extraction
```
for arch in R18 WRN28X10
do
for out_data in SVHN LSUN iSUN Texture places365
do
CUDA_VISIBLE_DEVICES=0 python feat_extract_ensemble.py \
 --arch ${arch} --in_data CIFAR10 --out_data ${out_data} \
 --model_path "./save/CIFAR10/${arch}/seed-1000/epoch150.pth" \
 "./save/CIFAR10/${arch}/seed-1200/epoch150.pth" \
 "./save/CIFAR10/${arch}/seed-2000/epoch150.pth"
done
done
```

2. Perform nearest neighbor search
```
for arch in R18 WRN28X10
do
CUDA_VISIBLE_DEVICES=0 python knn_ensemble.py \
 --arch ${arch} --in_data CIFAR10 --train_seed 1000 1200 2000
done
```

## Evaluation on the Mahalanobis detector

### Mahalanobis on single modes

To evaluate the Mahalanobis detector on single modes, the steps are:
1. Tuning hyper-parameters
```
cd utils_mahalanobis
for seed in 1000 1100 1200 1300 1400 2000 2100 2200 2300 2400
do
for arch in R18 WRN28X10 
do
CUDA_VISIBLE_DEVICES=0 python tune_mahalanobis_hyperparameter.py \
 --dataset CIFAR10 --data_dir YOUR_CIFAR10_DIR \
 --arch ${arch} --model_path "../save/CIFAR10/${arch}/seed-${seed}/epoch150.pth"
done
done
```
```
cd utils_mahalanobis
for seed in 1000 2000 3000 4000 5000
do
for arch in R50 DN121
do
CUDA_VISIBLE_DEVICES=0 python tune_mahalanobis_hyperparameter.py \
 --dataset ImageNet --data_dir YOUR_IMAGENET_DIR \
 --arch ${arch} --model_path "../save/ImageNet/${arch}/seed-${seed}/checkpoint.pth.tar" --batch_size 64 # 128 # 512
done
done
```

2. Run Mahalanobis detector
```
for seed in 1000 1100 1200 1300 1400 2000 2100 2200 2300 2400
do
for arch in R18 WRN28X10
do
for out_data in SVHN LSUN iSUN Texture places365
do
CUDA_VISIBLE_DEVICES=0 python eval_ood.py \
 --arch ${arch} --score Mahalanobis \
 --in_data CIFAR10 --out_data ${out_data} \
 --model_path "./save/CIFAR10/${arch}/seed-${seed}/epoch150.pth"
done
done
done
```
```
for seed in 1000 2000 3000 4000 5000
do
for arch in R50 DN121
do
for out_data in iNaturalist SUN Places Texture
do
CUDA_VISIBLE_DEVICES=1 python eval_ood.py \
 --arch ${arch} --score Mahalanobis \
 --in_data ImageNet --out_data ${out_data} --batch_size 64 \
 --model_path "./save/ImageNet/${arch}/seed-${seed}/checkpoint.pth.tar"
done
done
done
```

### Mahalanobis on mode ensemble

To evaluate the Mahalanobis detector on ensembling modes, the steps are:
1. Tuning hyper-parameters
```
cd utils_mahalanobis
for arch in R18 WRN28X10
do
CUDA_VISIBLE_DEVICES=0 python tune_mahalanobis_hyperparameter_ensemble.py \
 --arch ${arch} --dataset CIFAR10 --data_dir YOUR_CIFAR10_DIR --batch_size 32 \
 --model_path "./save/CIFAR10/${arch}/seed-1000/epoch150.pth" \
 "./save/CIFAR10/${arch}/seed-1200/epoch150.pth" \
 "./save/CIFAR10/${arch}/seed-2000/epoch150.pth"
done
```
```
cd utils_mahalanobis
for arch in R50 DN121
do
CUDA_VISIBLE_DEVICES=0 python tune_mahalanobis_hyperparameter_ensemble.py \
 --arch ${arch} --dataset ImageNet --data_dir YOUR_IMAGENET_DIR --batch_size 512 \
 --model_path "./save/ImageNet/${arch}/seed-1000/checkpoint.pth.tar" \
 "./save/ImageNet/${arch}/seed-3000/checkpoint.pth.tar" \
 "./save/ImageNet/${arch}/seed-5000/checkpoint.pth.tar"
done
```

2. Run Mahalanobis detector
```
for arch in R18 WRN28X10
do
for out_data in SVHN LSUN iSUN Texture places365
do
CUDA_VISIBLE_DEVICES=0 python eval_ood_ensemble.py \
 --arch ${arch} --score Mahalanobis --in_data CIFAR10 --out_data ${out_data} \
 --model_path "./save/CIFAR10/${arch}/seed-1000/epoch150.pth" \
 "./save/CIFAR10/${arch}/seed-1200/epoch150.pth" \
 "./save/CIFAR10/${arch}/seed-2000/epoch150.pth"
done
done
```
```
for arch in R50 DN121
do
for out_data in iNaturalist SUN Places Texture
do
CUDA_VISIBLE_DEVICES=1 python eval_ood_ensemble.py \
 --arch ${arch} --score Mahalanobis \
 --in_data ImageNet --out_data ${out_data} --batch_size 16 \
 --model_path "./save/ImageNet/${arch}/seed-1000/checkpoint.pth.tar" \
 "./save/ImageNet/${arch}/seed-3000/checkpoint.pth.tar" \
 "./save/ImageNet/${arch}/seed-5000/checkpoint.pth.tar"
done
done
```