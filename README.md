# ConVision Benchmark: A Contemporary Framework to Benchmark CNN and ViT Models
 
 🧑‍💻 **Authors:** Shreyas Bangalore Vijayakumar, Krishna Teja Chitty-Venkata, Kanishk Arya and Arun K. Somani

🏣 **Affiliation:** Iowa State University, Ames, IA

This repository is the official implementation of ["ConVision Benchmark"](https://www.mdpi.com/2673-2688/5/3/56) paper 




## Table of Contents

- [About](#-about)
- [Citation](#-citation)
- [Features](#features)
- [CNN Models](#cnn-models)
- [ViT Models](#vit-models)
- [Getting Started](#getting-started)
- [Accuracy Results](#accuracy-results-on-covid-19-dataset)
- [Computational Results](#computational-results-on-covid-19-dataset)
- [Acknowledgement](#acknowledgement)











## 📌 About
Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) have shown remarkable performance in computer vision tasks, including object detection and image recognition. These models have evolved significantly in architecture, efficiency, and versatility. Concurrently, deep-learning frameworks have diversified, with versions that often complicate reproducibility and unified benchmarking. We propose ConVision Benchmark, a comprehensive framework in PyTorch, to standardize the implementation and evaluation of state-of-the-art CNN and ViT models. This framework addresses common challenges such as version mismatches and inconsistent validation metrics. As a proof of concept, we performed an extensive benchmark analysis on a COVID-19 dataset, encompassing nearly 200 CNN and ViT models in which DenseNet-161 and MaxViT-Tiny achieved exceptional accuracy with a peak performance of around 95%. Although we primarily used the COVID-19 dataset for image classification, the framework is adaptable to a variety of datasets, enhancing its applicability across different domains. Our methodology includes rigorous performance evaluations, highlighting metrics such as accuracy, precision, recall, F1 score, and computational efficiency (FLOPs, MACs, CPU, and GPU latency). The ConVision Benchmark facilitates a comprehensive understanding of model efficacy, aiding researchers in deploying high-performance models for diverse applications.



## 📌 Citation
If you find this repository useful, please consider citing our paper:

```
@article{bangalore2024convision,
  title={ConVision Benchmark: A Contemporary Framework to Benchmark CNN and ViT Models},
  author={Bangalore Vijayakumar, Shreyas and Chitty-Venkata, Krishna Teja and Arya, Kanishk and Somani, Arun K},
  journal={AI},
  volume={5},
  number={3},
  pages={1132--1171},
  year={2024},
  publisher={MDPI}
}
```

## Features


We implemented CNN and ViT models in such a way a single model file contains the entire description of model without having to import any extra module (except torch and torchvision). For example, efficientnet_b0.py file contains the full code for Efficientnet_b0 model. One can directly use the file for any Computer vision application.  


We referred to several open source repositories for [CNN](https://github.com/pytorch/vision/tree/main/torchvision/models) and [ViT](https://github.com/lucidrains/vit-pytorch) models 


## CNN Models

| Models |   |
|-----------------------------------------------------------------------------------|----------------------------------------|
| [AlexNet](Models/CNN_Models/AlexNet) | [ConvNext_family](Models/CNN_Models/ConvNext_family) |
| [DenseNet_family](Models/CNN_Models/DenseNet_family) | [EfficientNet_family](Models/CNN_Models/EfficientNet_family) |
| [GhostNet_family](Models/CNN_Models/GhostNet_family) | [Inception_Family](Models/CNN_Models/Inception_Family) |
| [MNASNet_family](Models/CNN_Models/MNASNet_family) | [Mobilenet_family](Models/CNN_Models/Mobilenet_family) |
| [NFNet_Family](Models/CNN_Models/NFNet_Family) | [RegNet_family](Models/CNN_Models/RegNet_family) |
| [ResNet_family](Models/CNN_Models/ResNet_family) | [Shufflenet_family](Models/CNN_Models/Shufflenet_family) |
| [Squeezenet_family](Models/CNN_Models/Squeezenet_family) | [Vgg_family](Models/CNN_Models/Vgg_family) |



## ViT Models
| Models |   |
|-----------------------------------------------------------------------------------|----------------------------------------|
| [BoTNet_Family](Models/ViT_Models/BoTNet_Family) | [CCT_family](Models/ViT_Models/CCT_family) |
| [CaiT_family](Models/ViT_Models/CaiT_family) | [CrossFormer_family](Models/ViT_Models/CrossFormer_family) |
| [CrossViT_family](Models/ViT_Models/CrossViT_family) | [CvT_family](Models/ViT_Models/CvT_family) |
| [DeepViT_family](Models/ViT_Models/DeepViT_family) | [EdgeNeXt_Family](Models/ViT_Models/EdgeNeXt_Family) |
| [Efficientformer_family](Models/ViT_Models/Efficientformer_family) | [FocalTransformer_Family](Models/ViT_Models/FocalTransformer_Family) |
| [GC_ViT_Family](Models/ViT_Models/GC_ViT_Family) | [LVT_Family](Models/ViT_Models/LVT_Family) |
| [LeViT_family](Models/ViT_Models/LeViT_family) | [MLP_Mixer_Family](Models/ViT_Models/MLP_Mixer_Family) |
| [Max_ViT_Family](Models/ViT_Models/Max_ViT_Family) | [MobileFormer_Family](Models/ViT_Models/MobileFormer_Family) |
| [PVT_Family](Models/ViT_Models/PVT_Family) | [PiT_family](Models/ViT_Models/PiT_family) |
| [PoolFormer_Family](Models/ViT_Models/PoolFormer_Family) | [Region_ViT_family](Models/ViT_Models/Region_ViT_family) |
| [SepViT_family](Models/ViT_Models/SepViT_family) | [Swin_family](Models/ViT_Models/Swin_family) |
| [T2T_family](Models/ViT_Models/T2T_family) | [TNT_family](Models/ViT_Models/TNT_family) |
| [Twins_family](Models/ViT_Models/Twins_family) | [VAN_Family](Models/ViT_Models/VAN_Family) |
| [Vision_transformer](Models/ViT_Models/Vision_transformer) | | 



## Getting Started

Create a Virtual Environment
```bash
conda create -n ConVision python=3.11
conda activate ConVision
```

Install Requirements
```bash
pip install -r requirements.txt 
```

Curate Dataset
```bash
├── Dataset

│   ├── Train

│   ├──   ├── Class 1
│   ├──   ├──   ├── Image 1
│   ├──   ├──   ├── Image 2
│   ├──   ├──   ├── Image n

│   ├──   ├── Class 2
│   ├──   ├──   ├── Image 1
│   ├──   ├──   ├── Image 2
│   ├──   ├──   ├── Image n

│   ├──   ├── Class m
│   ├──   ├──   ├── Image 1
│   ├──   ├──   ├── Image 2
│   ├──   ├──   ├── Image n


│   ├── Test
│   ├──   ├── Class 1
│   ├──   ├──   ├── Image 1
│   ├──   ├──   ├── Image 2
│   ├──   ├──   ├── Image n

│   ├──   ├── Class 2
│   ├──   ├──   ├── Image 1
│   ├──   ├──   ├── Image 2
│   ├──   ├──   ├── Image n

│   ├──   ├── Class m
│   ├──   ├──   ├── Image 1
│   ├──   ├──   ├── Image 2
│   ├──   ├──   ├── Image n


│   ├── Val
│   ├──   ├── Class 1
│   ├──   ├──   ├── Image 1
│   ├──   ├──   ├── Image 2
│   ├──   ├──   ├── Image n

│   ├──   ├── Class 2
│   ├──   ├──   ├── Image 1
│   ├──   ├──   ├── Image 2
│   ├──   ├──   ├── Image n

│   ├──   ├── Class m
│   ├──   ├──   ├── Image 1
│   ├──   ├──   ├── Image 2
│   ├──   ├──   ├── Image n


```

Example to run ***ResNet18*** from ***ResNet_family*** in ***CNN_Models***  
```bash
python train/train.py \
            --model_name 'ResNet18' \
            --model_family 'ResNet_family' \
            --model_type 'CNN_Models' \
            --using_bn \
            --port 29500 \
            --model_dir '' \
            --train_root '' \
            --test_root '' \
            --val_root '' \
            --epochs 500 \
            --batch_size 256 \
            --workers 3 \
            --lr_mode 'cosine' \
            --base_lr 0.1 \
            --warmup_epochs 25 \
            --warmup_lr 0.0 \
            --targetlr 0.0 \
            --momentum 0.9 \
            --weight_decay 0.00005 \
            --using_moving_average \
            --last_gamma \
            --print_freq 100 \
            --evaluate \
```




## Accuracy Results on COVID-19 Dataset

| Model    | Top-1 Accuracy (\%) | Top-1 Accuracy Epoch | Recall (\%) | Recall Epoch | Best Loss | Best Loss Epoch |
|--------------------|------------------------------|-------------------------------|----------------------|-----------------------|--------------------|--------------------------|
| DenseNet-161       | 95.61                        | 410                           | 0.96                 | 410                   | 0.21               | 226                      |
| Vgg-13-bn          | 95.27                        | 230                           | 0.95                 | 230                   | 0.16               | 39                       |
| DenseNet-121       | 95.26                        | 401                           | 0.95                 | 401                   | 0.23               | 204                      |
| DenseNet-169       | 95.24                        | 327                           | 0.95                 | 327                   | 0.2                | 12                       |
| Vgg-19-bn          | 95.08                        | 187                           | 0.95                 | 187                   | 0.18               | 24                       |
| MaxVit-tiny        | 95.02                        | 296                           | 0.95                 | 296                   | 0.17               | 26                       |
| Vgg-11-bn          | 95.01                        | 293                           | 0.95                 | 293                   | 0.17               | 27                       |
| Vgg-16-bn          | 94.99                        | 131                           | 0.95                 | 131                   | 0.2                | 28                       |
| GoogLeNet          | 94.95                        | 136                           | 0.95                 | 136                   | 0.25               | 16                       |
| DenseNet-201       | 94.9                         | 383                           | 0.95                 | 383                   | 0.24               | 11                       |
| Vgg-16             | 94.27                        | 130                           | 0.94                 | 130                   | 0.21               | 20                       |
| EfficientNet-b4    | 94.25                        | 290                           | 0.94                 | 290                   | 0.19               | 30                       |
| MobileFormer-508M  | 94.15                        | 129                           | 0.94                 | 129                   | 0.25               | 24                       |
| MobileFormer-294M  | 94.11                        | 125                           | 0.94                 | 125                   | 0.26               | 9                        |
| Vgg-11             | 94.08                        | 154                           | 0.94                 | 154                   | 0.22               | 12                       |
| EfficientNet-b0    | 94.03                        | 253                           | 0.94                 | 253                   | 0.2                | 26                       |
| Vgg-13             | 94                           | 90                            | 0.94                 | 90                    | 0.22               | 11                       |
| MobileNetV2        | 93.97                        | 442                           | 0.94                 | 442                   | 0.26               | 17                       |
| Vgg-19             | 93.89                        | 176                           | 0.94                 | 176                   | 0.2                | 17                       |
| EfficientNet-b1    | 93.74                        | 371                           | 0.94                 | 371                   | 0.22               | 31                       |
| EfficientNet-b3    | 93.64                        | 154                           | 0.94                 | 154                   | 0.21               | 31                       |
| MobileFormer-151M  | 93.62                        | 134                           | 0.94                 | 134                   | 0.23               | 16                       |
| AlexNet            | 93.62                        | 267                           | 0.94                 | 267                   | 0.22               | 19                       |
| MobileFormer-214M  | 93.58                        | 444                           | 0.94                 | 444                   | 0.25               | 22                       |
| EfficientNet-v2-s  | 93.41                        | 361                           | 0.93                 | 361                   | 0.22               | 28                       |
| CCT-14-sine        | 93.3                         | 120                           | 0.93                 | 120                   | 0.24               | 33                       |
| MobileNetV1        | 93.27                        | 154                           | 0.93                 | 154                   | 0.3                | 460                      |
| MobileFormer-96M   | 93.21                        | 89                            | 0.93                 | 89                    | 0.3                | 26                       |
| CCT-7-sine         | 93.12                        | 170                           | 0.93                 | 170                   | 0.27               | 30                       |
| ResNet18           | 93.05                        | 170                           | 0.93                 | 170                   | 0.29               | 443                      |
| EfficientNet-b2    | 93                           | 257                           | 0.93                 | 257                   | 0.27               | 26                       |
| RegNet-y-16gf      | 93                           | 45                            | 0.93                 | 45                    | 0.31               | 11                       |
| CCT-14             | 92.9                         | 206                           | 0.93                 | 206                   | 0.24               | 26                       |
| EfficientNet-b5    | 92.87                        | 382                           | 0.93                 | 382                   | 0.26               | 38                       |
| ShuffleNetV2-x0-5  | 92.86                        | 278                           | 0.93                 | 278                   | 0.23               | 25                       |
| GhostNetV2         | 92.81                        | 160                           | 0.93                 | 160                   | 0.29               | 9                        |
| ResNet34           | 92.81                        | 271                           | 0.93                 | 271                   | 0.3                | 8                        |
| ShuffleNetV2-x2-0  | 92.78                        | 395                           | 0.93                 | 395                   | 0.31               | 12                       |
| RegNet-y-800mf     | 92.77                        | 58                            | 0.93                 | 58                    | 0.31               | 16                       |
| RegNet-y-1-6gf     | 92.75                        | 61                            | 0.93                 | 61                    | 0.25               | 14                       |
| ShuffleNetV2-x1-5  | 92.75                        | 236                           | 0.93                 | 236                   | 0.31               | 5                        |
| RegNet-y-8gf       | 92.72                        | 41                            | 0.93                 | 41                    | 0.29               | 15                       |
| RegNet-y-3-2gf     | 92.72                        | 68                            | 0.93                 | 68                    | 0.31               | 11                       |
| ShuffleNetV2-x1-0  | 92.6                         | 488                           | 0.93                 | 447                   | 0.28               | 15                       |
| MobileFormer-26M   | 92.58                        | 95                            | 0.93                 | 95                    | 0.26               | 18                       |
| ResNext50          | 92.53                        | 58                            | 0.93                 | 58                    | 0.28               | 10                       |
| MobileNet-V3-large | 92.47                        | 422                           | 0.92                 | 422                   | 0.36               | 61                       |
| PvT-v2-b3          | 92.43                        | 56                            | 0.92                 | 56                    | 0.24               | 19                       |
| RegNet-x-8gf       | 92.43                        | 218                           | 0.92                 | 218                   | 0.32               | 5                        |
| RegNet-x-1-6gf     | 92.38                        | 254                           | 0.92                 | 254                   | 0.33               | 467                      |
| RegNet-y-32gf      | 92.34                        | 64                            | 0.92                 | 64                    | 0.3                | 13                       |
| MobileFormer-52M   | 92.34                        | 135                           | 0.92                 | 135                   | 0.26               | 16                       |
| RegNet-y-400mf     | 92.32                        | 110                           | 0.92                 | 110                   | 0.3                | 23                       |
| RegNet-x-16gf      | 92.24                        | 434                           | 0.92                 | 434                   | 0.3                | 8                        |
| ResNet50           | 92.15                        | 233                           | 0.92                 | 233                   | 0.31               | 19                       |
| MNASnet-05         | 92.15                        | 499                           | 0.92                 | 499                   | 0.39               | 499                      |
| RegNet-x-400mf     | 92.13                        | 437                           | 0.92                 | 437                   | 0.33               | 22                       |
| RegNet-x-32gf          | 92.09                        | 488                           | 0.92                 | 488                   | 0.32               | 452                      |
| RegNet-x-3-2gf         | 92.02                        | 107                           | 0.92                 | 92                    | 0.32               | 15                       |
| FocalTransformer-Tiny  | 91.99                        | 392                           | 0.92                 | 392                   | 0.28               | 32                       |
| MNASnet-13             | 91.99                        | 499                           | 0.92                 | 499                   | 0.45               | 499                      |
| PvT-v2-b2              | 91.97                        | 121                           | 0.92                 | 121                   | 0.25               | 16                       |
| CCT-7                  | 91.96                        | 78                            | 0.92                 | 78                    | 0.27               | 28                       |
| Swin-ViT-Tiny-window7  | 91.84                        | 378                           | 0.92                 | 378                   | 0.28               | 24                       |
| MNASnet-075            | 91.82                        | 499                           | 0.92                 | 499                   | 0.46               | 499                      |
| RegNet-x-800mf         | 91.81                        | 401                           | 0.92                 | 401                   | 0.33               | 14                       |
| FocalTransformer-Small | 91.76                        | 269                           | 0.92                 | 269                   | 0.29               | 35                       |
| EfficientNet-v2-m      | 91.74                        | 450                           | 0.92                 | 450                   | 0.31               | 38                       |
| wide-ResNet50          | 91.74                        | 286                           | 0.92                 | 286                   | 0.27               | 13                       |
| mobilenet-v3-small     | 91.74                        | 469                           | 0.92                 | 469                   | 0.39               | 34                       |
| PvT-v2-b4              | 91.63                        | 80                            | 0.92                 | 80                    | 0.26               | 46                       |
| GCViT-xxTiny           | 91.62                        | 349                           | 0.92                 | 349                   | 0.26               | 29                       |
| Swin-ViT-Base          | 91.6                         | 340                           | 0.92                 | 340                   | 0.27               | 31                       |
| MNASnet-10             | 91.56                        | 499                           | 0.92                 | 499                   | 0.42               | 499                      |
| Swin-ViT-Small-window7 | 91.56                        | 369                           | 0.92                 | 369                   | 0.28               | 33                       |
| PvT-v2-b5              | 91.48                        | 46                            | 0.91                 | 46                    | 0.25               | 19                       |
| ResNet101              | 91.44                        | 481                           | 0.91                 | 481                   | 0.32               | 14                       |
| DeepViT-S              | 91.44                        | 340                           | 0.91                 | 340                   | 0.27               | 32                       |
| Swin-ViT-Large-window7 | 91.37                        | 402                           | 0.91                 | 402                   | 0.29               | 33                       |
| Swin-ViT-Small         | 91.34                        | 437                           | 0.91                 | 437                   | 0.28               | 31                       |
| PvT-v2-b2-Linear       | 91.32                        | 54                            | 0.91                 | 54                    | 0.29               | 26                       |
| VAN-b0                 | 91.32                        | 257                           | 0.91                 | 257                   | 0.28               | 238                      |
| PvT-v2-b1              | 91.32                        | 35                            | 0.91                 | 35                    | 0.25               | 22                       |
| GCViT-xTiny            | 91.28                        | 169                           | 0.91                 | 169                   | 0.26               | 33                       |
| ResNext101             | 91.25                        | 358                           | 0.91                 | 358                   | 0.33               | 6                        |
| Swin-ViT-Base-window7  | 91.15                        | 407                           | 0.91                 | 407                   | 0.29               | 34                       |
| T2T-ViT-T-24           | 91.12                        | 130                           | 0.91                 | 130                   | 0.3                | 32                       |
| Swin-ViT-Tiny          | 91.07                        | 389                           | 0.91                 | 389                   | 0.26               | 29                       |
| T2T-ViT-19             | 90.95                        | 169                           | 0.91                 | 169                   | 0.3                | 32                       |
| GCViT-Tiny             | 90.93                        | 111                           | 0.91                 | 111                   | 0.27               | 24                       |
| wide-ResNet101         | 90.9                         | 456                           | 0.91                 | 456                   | 0.4                | 5                        |
| GCViT-Tiny2            | 90.87                        | 81                            | 0.91                 | 81                    | 0.25               | 31                       |
| ResNet152              | 90.81                        | 108                           | 0.91                 | 108                   | 0.37               | 17                       |
| T2T-ViT-14-wide        | 90.79                        | 121                           | 0.91                 | 114                   | 0.32               | 15                       |
| CrossFormer-small      | 90.72                        | 232                           | 0.91                 | 232                   | 0.29               | 24                       |
| T2T-ViT-14             | 90.69                        | 133                           | 0.91                 | 133                   | 0.29               | 31                       |
| LVT                    | 90.65                        | 60                            | 0.91                 | 60                    | 0.28               | 28                       |
| NFNet-F0               | 90.65                        | 20                            | 0.91                 | 20                    | 0.27               | 20                       |
| CrossFormer-base       | 90.63                        | 332                           | 0.91                 | 332                   | 0.28               | 40                       |
| PvT-v2-b0              | 90.62                        | 71                            | 0.91                 | 71                    | 0.28               | 24                       |
| CrossFormer-large      | 90.6                         | 192                           | 0.91                 | 192                   | 0.27               | 30                       |
| DeepViT-L              | 90.54                        | 214                           | 0.91                 | 214                   | 0.3                | 87                       |
| CrossFormer-tiny       | 90.54                        | 419                           | 0.91                 | 419                   | 0.3                | 31                       |
| PvT-Large              | 90.48                        | 150                           | 0.9                  | 150                   | 0.31               | 32                       |
| T2T-ViT-10             | 90.34                        | 245                           | 0.9                  | 245                   | 0.3                | 32                       |
| RegionViT-Small        | 90.32                        | 135                           | 0.9                  | 135                   | 0.29               | 15                       |
| T2T-ViT-14-resnext     | 90.26                        | 236                           | 0.9                  | 236                   | 0.29               | 26                       |
| VAN-b1                 | 90.17                        | 64                            | 0.9                  | 64                    | 0.31               | 11                       |
| Twins-PCPVT-Large      | 90.11                        | 58                            | 0.9                  | 58                    | 0.28               | 21                       |
| Twins-SVT-Base         | 90.04                        | 52                            | 0.9                  | 52                    | 0.31               | 18                       |
| Sep-ViT-Small          | 90.03                        | 71                            | 0.9                  | 71                    | 0.31               | 25                       |
| VAN-b2                 | 89.97                        | 43                            | 0.9                  | 43                    | 0.33               | 14                       |
| T2T-ViT-24             | 89.95                        | 221                           | 0.9                  | 210                   | 0.32               | 28                       |
| Sep-ViT-Base           | 89.85                        | 104                           | 0.9                  | 104                   | 0.28               | 22                       |
| Twins-PCPVT-Base       | 89.81                        | 57                            | 0.9                  | 57                    | 0.31               | 18                       |
| T2T-ViT-7              | 89.81                        | 191                           | 0.9                  | 191                   | 0.29               | 39                       |
| PvT-Tiny               | 89.79                        | 153                           | 0.9                  | 153                   | 0.32               | 36                       |
| RegionViT-Base         | 89.7                         | 140                           | 0.9                  | 140                   | 0.3                | 25                       |
| Twins-SVT-Large        | 89.7                         | 63                            | 0.9                  | 63                    | 0.3                | 13                       |
| Sep-ViT-Tiny           | 89.7                         | 56                            | 0.9                  | 56                    | 0.29               | 23                       |
| PvT-Small             | 89.67                        | 87                            | 0.9                  | 87                    | 0.32               | 32                       |
| RegionViT-Medium      | 89.64                        | 179                           | 0.9                  | 179                   | 0.3                | 17                       |
| T2T-ViT-T-19          | 89.58                        | 104                           | 0.9                  | 104                   | 0.32               | 24                       |
| T2T-ViT-12            | 89.54                        | 229                           | 0.9                  | 229                   | 0.29               | 33                       |
| PiT-XS                | 89.5                         | 97                            | 0.89                 | 97                    | 0.33               | 21                       |
| RegionViT-Tiny        | 89.47                        | 157                           | 0.89                 | 157                   | 0.32               | 19                       |
| TNT-Base              | 89.41                        | 215                           | 0.89                 | 215                   | 0.32               | 38                       |
| Twins-SVT-Small       | 89.39                        | 57                            | 0.89                 | 57                    | 0.31               | 21                       |
| T2T-ViT-T-14          | 89.39                        | 134                           | 0.89                 | 134                   | 0.31               | 31                       |
| Twins-PCPVT-Small     | 89.3                         | 49                            | 0.89                 | 49                    | 0.29               | 18                       |
| CrossViT-15-dagger    | 89.25                        | 236                           | 0.89                 | 236                   | 0.32               | 31                       |
| PiT-Small             | 89.11                        | 103                           | 0.89                 | 103                   | 0.34               | 18                       |
| CrossViT-9-dagger     | 89.04                        | 135                           | 0.89                 | 135                   | 0.33               | 34                       |
| CvT-21                | 89.04                        | 286                           | 0.89                 | 286                   | 0.37               | 24                       |
| CrossViT-Small        | 88.85                        | 254                           | 0.89                 | 254                   | 0.32               | 41                       |
| CrossViT-15           | 88.77                        | 207                           | 0.89                 | 207                   | 0.33               | 36                       |
| PiT-TI                | 88.74                        | 120                           | 0.89                 | 120                   | 0.35               | 16                       |
| CrossViT-Base         | 88.74                        | 179                           | 0.89                 | 179                   | 0.34               | 33                       |
| EdgeNeXt-BNHS-Xsmall  | 88.74                        | 54                            | 0.89                 | 54                    | 0.3                | 38                       |
| TNT-Small             | 88.72                        | 171                           | 0.89                 | 171                   | 0.31               | 45                       |
| ConvNext-Small        | 88.67                        | 59                            | 0.89                 | 59                    | 0.32               | 59                       |
| CrossViT-18           | 88.57                        | 265                           | 0.89                 | 265                   | 0.33               | 41                       |
| ViT-Small-patch8      | 88.41                        | 43                            | 0.88                 | 43                    | 0.32               | 43                       |
| Sep-ViT-Lite          | 88.35                        | 27                            | 0.88                 | 27                    | 0.32               | 27                       |
| ViT-Small-patch16     | 88.29                        | 204                           | 0.88                 | 204                   | 0.34               | 37                       |
| EdgeNeXt-BNHS-Small   | 88.27                        | 253                           | 0.88                 | 253                   | 0.35               | 42                       |
| CvT-13                | 88.21                        | 68                            | 0.88                 | 68                    | 0.35               | 27                       |
| CaiT-XS24             | 88.16                        | 108                           | 0.88                 | 108                   | 0.33               | 80                       |
| CrossViT-Tiny         | 87.99                        | 218                           | 0.88                 | 218                   | 0.32               | 46                       |
| CaiT-XS36             | 87.95                        | 84                            | 0.88                 | 84                    | 0.33               | 72                       |
| CrossViT-9            | 87.82                        | 200                           | 0.88                 | 200                   | 0.33               | 41                       |
| ViT-Tiny-patch16      | 87.67                        | 182                           | 0.88                 | 182                   | 0.36               | 39                       |
| CaiT-XXS24            | 87.51                        | 102                           | 0.88                 | 102                   | 0.35               | 88                       |
| ConvNext-Tiny         | 87.48                        | 68                            | 0.87                 | 68                    | 0.35               | 68                       |
| CaiT-S24              | 87.46                        | 74                            | 0.87                 | 74                    | 0.34               | 59                       |
| MLPMixer              | 87.43                        | 60                            | 0.87                 | 60                    | 0.37               | 6                        |
| ViT-Base-patch16      | 87.29                        | 186                           | 0.87                 | 186                   | 0.36               | 34                       |
| ConvNext-Base         | 87.24                        | 47                            | 0.87                 | 47                    | 0.35               | 47                       |
| ResMLP                | 87.11                        | 38                            | 0.87                 | 38                    | 0.36               | 31                       |
| EdgeNeXt-Small        | 87.09                        | 69                            | 0.87                 | 69                    | 0.35               | 54                       |
| CaiT-XXS36            | 87.05                        | 74                            | 0.87                 | 74                    | 0.35               | 74                       |
| EdgeNeXt-Base         | 86.93                        | 54                            | 0.87                 | 54                    | 0.36               | 46                       |
| ViT-Large-patch32     | 86.73                        | 167                           | 0.87                 | 167                   | 0.4                | 33                       |
| EdgeNeXt-Xsmall       | 86.65                        | 57                            | 0.87                 | 57                    | 0.37               | 42                       |
| ViT-Base-patch32      | 86.34                        | 155                           | 0.86                 | 154                   | 0.41               | 36                       |
| ViT-Small-patch32     | 86.06                        | 129                           | 0.86                 | 129                   | 0.43               | 25                       |
| EdgeNeXt-BNHS-Xxsmall | 85.92                        | 58                            | 0.86                 | 58                    | 0.4                | 38                       |
| GCViT-Small           | 85.75                        | 65                            | 0.86                 | 65                    | 0.39               | 32                       |
| PoolFormer-S36        | 85.22                        | 484                           | 0.85                 | 484                   | 0.39               | 442                      |
| SqueezeNet-1-1        | 85.11                        | 463                           | 0.63                 | 445                   | 0.59               | 437                      |
| PoolFormer-S12        | 83.91                        | 37                            | 0.84                 | 37                    | 0.42               | 37                       |
| NFNet-F1              | 83.44                        | 10                            | 0.83                 | 10                    | 0.44               | 10                       |
| BoTNet                | 83.38                        | 294                           | 0.83                 | 294                   | 0.45               | 151                      |
| PiT-Base              | 81.51                        | 20                            | 0.82                 | 20                    | 0.47               | 20                       |
| PoolFormer-M36        | 81.38                        | 20                            | 0.81                 | 20                    | 0.48               | 14                       |
| PoolFormer-S24        | 80.38                        | 38                            | 0.8                  | 38                    | 0.49               | 33                       |
| PvT-Medium            | 75.84                        | 4                             | 0.76                 | 4                     | 0.6                | 4                        |
| SqueezeNet-1-0        | 45.54                        | 2                             | 0.46                 | 9                     | 1.02               | 1                        |
| Inception-Resnet-v2   | 34.69                        | 18                            | 0.35                 | 18                    | 1.06               | 16                       |


## Computational Results on COVID-19 Dataset

| Model | Number of Parameters (Million) | MACs (billion) | FLOPs (billion) | Training Time per Epoch (s) | CPU Latency (ms) | GPU Latency (ms) | Training Memory (GB) | Inference Memory (MB) |
|------------------------|-----------------------------------------|-------------------------|--------------------------|--------------------------------------|---------------------------|---------------------------|-------------------------------|--------------------------------|
| CrossViT-9             | 8.07                                    | 1.54                    | 1.85                     | 28.48                                | 16.75                     | 7.16                      | 0.17                          | 31.28                          |
| RegNet-x-1-6gf         | 8.28                                    | 1.63                    | 1.62                     | 19.06                                | 17.01                     | 4.7                       | 0.19                          | 31.98                          |
| CrossViT-9-dagger      | 8.29                                    | 1.68                    | 1.99                     | 29.22                                | 17.61                     | 7.43                      | 0.18                          | 32.14                          |
| MobileFormer-294M      | 9.51                                    | 0.29                    | 0.29                     | 23.48                                | 21.14                     | 15.82                     | 0.21                          | 36.69                          |
| PiT-XS                 | 10.16                                   | 1.1                     | 1.1                      | 18.32                                | 12.31                     | 3.5                       | 0.18                          | 39.2                           |
| RegNet-y-1-6gf         | 10.32                                   | 1.65                    | 1.63                     | 21.78                                | 24.06                     | 10.79                     | 0.22                          | 39.87                          |
| EfficientNet-b3        | 10.69                                   | 0.97                    | 0.95                     | 36.41                                | 24.44                     | 8.72                      | 0.31                          | 41.52                          |
| ResNet18               | 11.18                                   | 1.82                    | 1.82                     | 17.69                                | 9.42                      | 1.57                      | 0.19                          | 42.7                           |
| PoolFormer-S12         | 11.38                                   | 1.81                    | 1.82                     | 17.48                                | 17.63                     | 3.36                      | 0.23                          | 44.42                          |
| GCViT-xxTiny           | 11.44                                   | 1.96                    | 2.14                     | 47.39                                | 27.75                     | 9.31                      | 0.28                          | 46.19                          |
| CaiT-XXS24             | 11.72                                   | 2.18                    | 2.53                     | 54.46                                | 37.67                     | 11.8                      | 0.28                          | 45.11                          |
| MobileFormer-508M      | 12.06                                   | 0.5                     | 0.51                     | 29.69                                | 26.1                      | 15.85                     | 0.28                          | 46.47                          |
| PvT-Tiny               | 12.33                                   | 1.86                    | 1.94                     | 27.91                                | 18.65                     | 4.34                      | 0.26                          | 48.7                           |
| DenseNet-169           | 12.49                                   | 3.43                    | 3.4                      | 35.04                                | 46.82                     | 14.04                     | 0.34                          | 48.48                          |
| RegionViT-Tiny         | 13.31                                   | 2.31                    | 2.43                     | 52.56                                | 31.93                     | 11.85                     | 0.3                           | 50.98                          |
| VAN-b1                 | 13.34                                   | 2.51                    | 2.5                      | 35.84                                | 29.37                     | 5.48                      | 0.32                          | 51.57                          |
| PvT-v2-b1              | 13.5                                    | 2.04                    | 2.12                     | 45.28                                | 25.29                     | 4.76                      | 0.29                          | 51.56                          |
| RegNet-x-3-2gf         | 14.29                                   | 3.22                    | 3.2                      | 24.6                                 | 27.22                     | 6.37                      | 0.3                           | 55.01                          |
| ResMLP                 | 14.94                                   | 3.01                    | 3.01                     | 32                                   | 14.76                     | 2.93                      | 0.27                          | 58.06                          |
| CaiT-XXS36             | 17.06                                   | 3.24                    | 3.77                     | 80.26                                | 57.31                     | 17.66                     | 0.42                          | 65.6                           |
| EfficientNet-b4        | 17.55                                   | 1.58                    | 1.54                     | 49.72                                | 32.27                     | 10.61                     | 0.48                          | 68.25                          |
| EdgeNeXt-Base          | 17.91                                   | 2.92                    | 2.95                     | 45.6                                 | 26.18                     | 5.54                      | 0.38                          | 69.99                          |
| RegNet-y-3-2gf         | 17.93                                   | 3.22                    | 3.2                      | 28.75                                | 30.39                     | 8.31                      | 0.37                          | 69.61                          |
| DenseNet-201           | 18.1                                    | 4.39                    | 4.34                     | 43.93                                | 61.95                     | 17.11                     | 0.46                          | 70.19                          |
| BoTNet                 | 18.8                                    | 4.02                    | 4.06                     | 23.98                                | 28.94                     | 4.86                      | 0.37                          | 72.1                           |
| GCViT-xTiny            | 19.42                                   | 2.71                    | 2.94                     | 62.8                                 | 37.31                     | 12.06                     | 0.43                          | 76.8                           |
| CvT-13                 | 19.61                                   | 4.08                    | 4.58                     | 58.61                                | 39.41                     | 11.54                     | 0.4                           | 75.07                          |
| EfficientNet-v2-s      | 20.18                                   | 2.9                     | 2.88                     | 32.62                                | 36.04                     | 11.8                      | 0.46                          | 79.03                          |
| PoolFormer-S24         | 20.84                                   | 3.39                    | 3.41                     | 23.6                                 | 33.95                     | 6.59                      | 0.42                          | 80.66                          |
| T2T-ViT-T-14           | 21.08                                   | 4.35                    | 6.11                     | 74.47                                | 38.4                      | 5.92                      | 0.44                          | 80.8                           |
| T2T-ViT-14             | 21.08                                   | 4.35                    | 4.8                      | 55.18                                | 28.44                     | 6.35                      | 0.4                           | 80.81                          |
| T2T-ViT-14-resnext     | 21.08                                   | 4.35                    | 4.8                      | 87.15                                | 30.72                     | 6.4                       | 0.46                          | 80.81                          |
| ResNet34               | 21.29                                   | 3.68                    | 3.67                     | 17.28                                | 17.02                     | 2.62                      | 0.35                          | 82.18                          |
| ViT-Small-patch8       | 21.37                                   | 16.76                   | 16.76                    | 184.67                               | 96.67                     | 4.8                       | 0.55                          | 83.08                          |
| ViT-Small-patch16      | 21.59                                   | 4.25                    | 4.25                     | 41.85                                | 23.29                     | 3.11                      | 0.38                          | 83.09                          |
| NFNet-F0               | 21.86                                   | 0.02                    | 9.18                     | 38.27                                | 105.73                    | 10.76                     | 1.69                          | 263.07                         |
| CCT-14-sine            | 21.91                                   | 5.12                    | 5.53                     | 50.1                                 | 27.48                     | 4.8                       | 0.4                           | 84.13                          |
| CCT-14                 | 21.91                                   | 5.12                    | 5.53                     | 50.08                                | 27.86                     | 4.89                      | 0.4                           | 84.13                          |
| PvT-v2-b2-Linear       | 22.04                                   | 3.76                    | 3.91                     | 87.54                                | 44.64                     | 10.11                     | 0.49                          | 85.04                          |
| ViT-Small-patch32      | 22.48                                   | 1.12                    | 1.12                     | 17.68                                | 10.95                     | 3.16                      | 0.35                          | 85.94                          |
| PiT-Small              | 22.78                                   | 2.42                    | 2.42                     | 31.43                                | 21.77                     | 3.65                      | 0.39                          | 87.48                          |
| ResNext50              | 22.99                                   | 4.29                    | 4.26                     | 28.87                                | 32.07                     | 4.12                      | 0.45                          | 88.61                          |
| TNT-Small              | 23.3                                    | 4.85                    | 5.24                     | 118.45                               | 43.93                     | 10.97                     | 0.49                          | 91.95                          |
| ResNet50               | 23.51                                   | 4.13                    | 4.11                     | 22.31                                | 27.27                     | 3.95                      | 0.43                          | 89.95                          |
| Twins-SVT-Small        | 23.55                                   | 2.82                    | 2.82                     | 42.83                                | 31.37                     | 8.28                      | 0.44                          | 90.98                          |
| PvT-Small              | 23.58                                   | 3.69                    | 3.83                     | 49.5                                 | 33.56                     | 8.27                      | 0.48                          | 91.69                          |
| Twins-PCPVT-Small      | 23.59                                   | 3.68                    | 3.68                     | 47.79                                | 36.44                     | 7.33                      | 0.47                          | 90.47                          |
| T2T-ViT-14-wide        | 24.23                                   | 4.97                    | 5.24                     | 52.51                                | 23.82                     | 3.17                      | 0.42                          | 93.39                          |
| PvT-v2-b2              | 24.85                                   | 3.9                     | 4.05                     | 80.47                                | 43.29                     | 9.24                      | 0.53                          | 94.93                          |
| VAN-b2                 | 26.06                                   | 5.01                    | 5                        | 64.12                                | 57.48                     | 10.47                     | 0.6                           | 100.61                         |
| CrossViT-Small         | 26.13                                   | 5.08                    | 5.63                     | 65                                   | 33.22                     | 7.91                      | 0.5                           | 100.72                         |
| CaiT-XS24              | 26.2                                    | 4.87                    | 5.4                      | 84.85                                | 59.42                     | 11.93                     | 0.56                          | 100.48                         |
| DenseNet-161           | 26.48                                   | 7.84                    | 7.78                     | 52.73                                | 73.24                     | 14.4                      | 0.63                          | 103.1                          |
| CrossViT-15-dagger     | 27.48                                   | 5.49                    | 6.13                     | 67.44                                | 37.02                     | 8.9                       | 0.53                          | 106.18                         |
| Swin-ViT-Tiny-window7  | 27.5                                    | 4.37                    | 4.51                     | 62.9                                 | 30.61                     | 6.26                      | 0.52                          | 106.45                         |
| Swin-ViT-Base-window7  | 27.5                                    | 4.37                    | 4.51                     | 62.94                                | 31.31                     | 6.45                      | 0.52                          | 106.45                         |
| Swin-ViT-Tiny          | 27.5                                    | 4.38                    | 4.64                     | 65.37                                | 32.72                     | 5.9                       | 0.54                          | 108.14                         |
| GCViT-Tiny             | 27.58                                   | 4.32                    | 4.79                     | 90.34                                | 59.13                     | 19.18                     | 0.62                          | 112.12                         |
| ConvNext-Tiny          | 27.81                                   | 4.46                    | 4.47                     | 60.66                                | 25.91                     | 3.65                      | 0.52                          | 107.11                         |
| EfficientNet-b5        | 28.35                                   | 2.46                    | 2.41                     | 67.67                                | 45                        | 12.76                     | 0.72                          | 110.67                         |
| FocalTransformer-Tiny  | 29.44                                   | 4.66                    | 5.22                     | 146.14                               | 65.75                     | 17.82                     | 0.62                          | 116                            |
| RegionViT-Small        | 29.79                                   | 5.19                    | 5.35                     | 84.44                                | 44.81                     | 11.9                      | 0.59                          | 114.75                         |
| CrossFormer-small      | 29.89                                   | 4.79                    | 4.92                     | 63.15                                | 37.44                     | 9.12                      | 0.56                          | 114.42                         |
| PoolFormer-S36         | 30.29                                   | 4.97                    | 5                        | 33.99                                | 50.58                     | 9.5                       | 0.62                          | 116.89                         |
| MaxVit-tiny            | 30.38                                   | 5.46                    | 5.61                     | 91.16                                | 64.36                     | 17.1                      | 0.75                          | 118.93                         |
| Sep-ViT-Tiny           | 30.4                                    | 4.28                    | 4.53                     | 60.46                                | 32.76                     | 6.82                      | 0.56                          | 116.46                         |
| CvT-21                 | 31.24                                   | 6.54                    | 7.21                     | 89.71                                | 60.54                     | 18.49                     | 0.63                          | 119.56                         |
| GCViT-Tiny2            | 33.84                                   | 5.56                    | 6.21                     | 110.91                               | 67.88                     | 23.66                     | 0.76                          | 139.19                         |
| RegNet-y-8gf           | 37.17                                   | 8.04                    | 8                        | 41.54                                | 49.33                     | 6.97                      | 0.71                          | 143.66                         |
| RegNet-x-8gf           | 37.66                                   | 8.05                    | 8.02                     | 33.09                                | 46.26                     | 6.53                      | 0.67                          | 144.42                         |
| CaiT-XS36              | 38.19                                   | 7.25                    | 8.05                     | 125.61                               | 87.81                     | 16.97                     | 0.81                          | 146.36                         |
| T2T-ViT-T-19           | 38.64                                   | 7.8                     | 9.81                     | 111.49                               | 54.7                      | 7.3                       | 0.75                          | 147.87                         |
| T2T-ViT-19             | 38.64                                   | 7.8                     | 8.5                      | 91.81                                | 43.65                     | 8.18                      | 0.71                          | 147.88                         |
| RegionViT-Medium       | 40.41                                   | 7.22                    | 7.43                     | 106.99                               | 55.75                     | 16.46                     | 0.76                          | 155.33                         |
| CrossViT-18            | 42.42                                   | 8.21                    | 9.05                     | 99.93                                | 48.94                     | 9.79                      | 0.78                          | 163.05                         |
| ResNet101              | 42.51                                   | 7.86                    | 7.83                     | 33.93                                | 49.66                     | 7.91                      | 0.75                          | 162.89                         |
| PvT-Medium             | 43.31                                   | 6.46                    | 6.69                     | 76.58                                | 53.59                     | 14.11                     | 0.82                          | 167.01                         |
| Twins-PCPVT-Base       | 43.32                                   | 6.46                    | 6.46                     | 72.79                                | 57.61                     | 12.48                     | 0.8                           | 165.82                         |
| NFNet-F1               | 43.7                                    | 0.04                    | 16.92                    | 66.52                                | 189.01                    | 20.18                     | 3.17                          | 500.84                         |
| PvT-v2-b3              | 44.73                                   | 6.7                     | 6.92                     | 116.41                               | 68.55                     | 15.47                     | 0.88                          | 170.85                         |
| Sep-ViT-Small          | 45.78                                   | 7.07                    | 7.48                     | 93.93                                | 50.93                     | 11.43                     | 0.84                          | 176.7                          |
| CaiT-S24               | 46.44                                   | 8.63                    | 9.35                     | 118.42                               | 80.53                     | 12.09                     | 0.91                          | 177.99                         |
| Swin-ViT-Small-window7 | 48.79                                   | 8.54                    | 8.77                     | 107.99                               | 56.78                     | 12.5                      | 0.9                           | 188.27                         |
| Swin-ViT-Large-window7 | 48.79                                   | 8.55                    | 8.77                     | 125.08                               | 58.6                      | 12.65                     | 0.92                          | 188.46                         |
| Swin-ViT-Small         | 48.79                                   | 8.58                    | 9.43                     | 121.63                               | 64.03                     | 10.84                     | 0.98                          | 196.6                          |
| ConvNext-Small         | 49.44                                   | 8.7                     | 8.7                      | 108.96                               | 45.36                     | 6.8                       | 0.91                          | 189.73                         |
| GCViT-Small            | 50.11                                   | 7.87                    | 8.57                     | 135.43                               | 73.2                      | 18.98                     | 1.05                          | 199.24                         |
| FocalTransformer-Small | 50.74                                   | 8.87                    | 9.75                     | 244.35                               | 115.78                    | 35.54                     | 1.07                          | 201.32                         |
| CrossFormer-base       | 51.2                                    | 8.96                    | 9.19                     | 107.93                               | 63.41                     | 17.06                     | 0.94                          | 196.1                          |
| RegNet-x-16gf          | 52.24                                   | 16.04                   | 15.99                    | 53.72                                | 74.41                     | 6.54                      | 0.98                          | 202.09                         |
| EfficientNet-v2-m      | 52.86                                   | 5.45                    | 5.41                     | 53.7                                 | 65.57                     | 16.43                     | 1.05                          | 207.29                         |
| Inception-Resnet-v2    | 54.31                                   | 6.5                     | 6.48                     | 39                                   | 82.18                     | 19.93                     | 0.91                          | 210.51                         |
| Twins-SVT-Base         | 55.3                                    | 8.36                    | 8.36                     | 92.75                                | 58.72                     | 11.19                     | 0.99                          | 214.47                         |
| PoolFormer-M36         | 55.32                                   | 8.76                    | 8.8                      | 48                                   | 75.19                     | 9.47                      | 1.07                          | 211.63                         |
| AlexNet                | 57.02                                   | 0.71                    | 0.71                     | 17.03                                | 8.95                      | 0.51                      | 0.85                          | 217.5                          |
| ResNet152              | 58.15                                   | 11.6                    | 11.56                    | 46.75                                | 72.23                     | 12.03                     | 1.04                          | 222.54                         |
| DeepViT-L              | 58.38                                   | 12.16                   | 13.19                    | 184.18                               | 82.44                     | 13.83                     | 1.12                          | 223.34                         |
| PvT-Large              | 60.47                                   | 9.53                    | 9.85                     | 109                                  | 79.53                     | 20.33                     | 1.14                          | 232.44                         |
| Twins-PCPVT-Large      | 60.48                                   | 9.52                    | 9.53                     | 102.84                               | 82.13                     | 18.06                     | 1.12                          | 231.77                         |
| PvT-v2-b4              | 62.04                                   | 9.82                    | 10.14                    | 165.58                               | 97.98                     | 23.3                      | 1.22                          | 237.76                         |
| T2T-ViT-T-24           | 63.49                                   | 12.7                    | 15                       | 154.36                               | 71.54                     | 9.34                      | 1.17                          | 243.59                         |
| T2T-ViT-24             | 63.49                                   | 12.7                    | 13.69                    | 134.95                               | 63.92                     | 10.35                     | 1.13                          | 243.6                          |
| TNT-Base               | 64.64                                   | 13.44                   | 14.09                    | 196.07                               | 80.53                     | 11.11                     | 1.19                          | 247.23                         |
| wide-ResNet50          | 66.84                                   | 11.45                   | 11.42                    | 33.16                                | 60.34                     | 4.02                      | 1.11                          | 257.28                         |
| Swin-ViT-Base          | 70.09                                   | 12.75                   | 13.69                    | 164.92                               | 89.29                     | 15.83                     | 1.36                          | 278.57                         |
| RegionViT-Base         | 71.67                                   | 12.79                   | 13.07                    | 161.49                               | 85.48                     | 16.52                     | 1.3                           | 276.61                         |
| PiT-Base               | 72.5                                    | 10.55                   | 10.56                    | 105.05                               | 59.57                     | 3.98                      | 1.22                          | 279.65                         |
| RegNet-y-16gf          | 80.57                                   | 16.01                   | 15.96                    | 58.87                                | 83.78                     | 8.36                      | 1.42                          | 316.53                         |
| Sep-ViT-Base           | 81.33                                   | 12.54                   | 13.08                    | 144.88                               | 73.92                     | 11.35                     | 1.42                          | 312.46                         |
| PvT-v2-b5              | 81.44                                   | 11.38                   | 11.76                    | 170.68                               | 109.8                     | 29.59                     | 1.5                           | 311.92                         |
| ViT-Base-patch16       | 85.65                                   | 16.86                   | 16.87                    | 136.99                               | 65.71                     | 4.78                      | 1.39                          | 327.43                         |
| ResNext101             | 86.75                                   | 16.54                   | 16.47                    | 69.47                                | 95.97                     | 7.93                      | 1.55                          | 336.15                         |
| ViT-Base-patch32       | 87.42                                   | 4.37                    | 4.37                     | 35.79                                | 27.77                     | 3.3                       | 1.33                          | 333.75                         |
| ConvNext-Base          | 87.55                                   | 15.37                   | 15.38                    | 171.72                               | 73.34                     | 6.8                       | 1.53                          | 334.24                         |
| CrossFormer-large      | 90.95                                   | 15.85                   | 16.15                    | 168.25                               | 90.3                      | 16.84                     | 1.61                          | 351.68                         |
| Twins-SVT-Large        | 98.25                                   | 14.83                   | 14.84                    | 146.98                               | 85.83                     | 11.1                      | 1.69                          | 379.97                         |
| CrossViT-Base          | 103.57                                  | 20.13                   | 21.22                    | 188.87                               | 90.13                     | 7.96                      | 1.79                          | 407.83                         |
| RegNet-x-32gf          | 105.3                                   | 31.88                   | 31.81                    | 100.28                               | 132.68                    | 6.82                      | 1.85                          | 402.73                         |
| wide-ResNet101         | 124.84                                  | 22.84                   | 22.79                    | 55.16                                | 107.95                    | 8.07                      | 2.05                          | 484.85                         |
| Vgg-11                 | 128.78                                  | 7.61                    | 7.61                     | 20.46                                | 40.37                     | 0.95                      | 1.97                          | 491.25                         |
| Vgg-11-bn              | 128.78                                  | 7.63                    | 7.62                     | 25.71                                | 40.52                     | 1.08                      | 1.99                          | 491.3                          |
| Vgg-13                 | 128.96                                  | 11.3                    | 11.3                     | 30.26                                | 49.86                     | 1.08                      | 1.99                          | 491.96                         |
| Vgg-13-bn              | 128.97                                  | 11.35                   | 11.33                    | 39.61                                | 50.98                     | 1.26                      | 2.03                          | 492.01                         |
| Vgg-16                 | 134.27                                  | 15.47                   | 15.47                    | 35.48                                | 60.87                     | 1.3                       | 2.07                          | 513.09                         |
| Vgg-16-bn              | 134.28                                  | 15.52                   | 15.49                    | 45.63                                | 62.06                     | 1.52                      | 2.12                          | 513.16                         |
| Vgg-19                 | 139.58                                  | 19.63                   | 19.63                    | 41.03                                | 73.43                     | 1.52                      | 2.16                          | 532.47                         |
| Vgg-19-bn              | 139.59                                  | 19.69                   | 19.66                    | 51.7                                 | 73.04                     | 1.71                      | 2.21                          | 532.56                         |
| RegNet-y-32gf          | 141.34                                  | 32.4                    | 32.34                    | 88.81                                | 146.35                    | 8.37                      | 2.41                          | 551.99                         |
| ViT-Large-patch32      | 305.46                                  | 15.26                   | 15.27                    | 114.15                               | 90.16                     | 6.2                       | 4.63                          | 1165.68                        |


## Acknowledgement