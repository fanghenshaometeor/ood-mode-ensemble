# Training and Evaluation commands

All the commands to reproduce the reported results in our paper are listed below.

## Training isolated modes on CIFAR10 and ImageNet

### CIFAR10

To train isolated modes on CIFAR10, run the following BASH command
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

To train isolated modes on ImageNet, run the following BASH command
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
 "./save/ImageNet/${arch}/seed-5000/checkpoint.pth.tar" \
done
done
```

## Evaluation on the kNN detector

## Evaluation on the Mahalanobis detector