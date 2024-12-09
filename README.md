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





## Accuracy Results on COVID-19 Dataset


<!-- | Model-Name | Best-Top-1 | Best-F1-score | Best-Loss | Best-Precision | Best-Recall | Best-FPR | Best-FNR | Best-MCC | MACs | FLOPS | Number-of-Parameters | CPU-latency | GPU-latency  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet18 |  |  |  |  |  |  |  | | |  |  |  |  | -->


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


## Computational Results on COVID-19 Dataset



## Acknowledgement