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
| AlexNet | ConvNext  | 
| DenseNet | EfficientNet |
| Ghost-resnet | GhostNetv2 | 
| Inception | MNASNet |
| MobileNet | NFNet |
| RegNet | ResNet |
| ResNext | Wide-ResNet | 
| ShuffleNetv2 | SqueezeNet |
| VGG | |


 [AlexNet](Models/CNN_Models/AlexNet/)

AlexNet
ConvNext
DenseNet
EfficientNet
Ghost-resnet
GhostNetv2
Inception
MNASNet
MobileNet
NFNet
RegNet
ResNet
ResNext | Wide-ResNet | ShuffleNetv2 | SqueezeNet  |
| VGG | | | |



## ViT Models
| Models |   |  |  |
|----------------------------------------------|---------------------------------------|--------------------------------------------------------------------|------------------------------------------|
| BoTNet       | CaiT  | CCT  | CrossFormer |
| CrossViT | CvT | DeepViT  | EdgeNeXt |
| EfficientFormer | FocalTransformer | GC-ViT | LeViT |
| LVT | Max-ViT | MLP-Mixer | MobileFormer |
| PiT  | PoolFormer | PVT  | Region-ViT  |
| SepViT | Swin | T2T-ViT  | TNT |
| Twins | VAN | Vision Transformer | | 


## Getting Started





## Accuracy Results on COVID-19 Dataset


<!-- | Model-Name | Best-Top-1 | Best-F1-score | Best-Loss | Best-Precision | Best-Recall | Best-FPR | Best-FNR | Best-MCC | MACs | FLOPS | Number-of-Parameters | CPU-latency | GPU-latency  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet18 |  |  |  |  |  |  |  | | |  |  |  |  | -->


| Model    | Top-1 Accuracy (\%) | Top-1 Accuracy Epoch | Recall (\%) | Recall Epoch | Best Loss | Best Loss Epoch |
|-------------------|------------------------------|-------------------------------|----------------------|-----------------------|--------------------|--------------------------|
| DenseNet-161      | 95.61                        | 410                           | 0.96                 | 410                   | 0.21               | 226                      |
| Vgg-13-bn         | 95.27                        | 230                           | 0.95                 | 230                   | 0.16               | 39                       |
| DenseNet-121      | 95.26                        | 401                           | 0.95                 | 401                   | 0.23               | 204                      |
| DenseNet-169      | 95.24                        | 327                           | 0.95                 | 327                   | 0.2                | 12                       |
| Vgg-19-bn         | 95.08                        | 187                           | 0.95                 | 187                   | 0.18               | 24                       |
| MaxVit-tiny       | 95.02                        | 296                           | 0.95                 | 296                   | 0.17               | 26                       |
| Vgg-11-bn         | 95.01                        | 293                           | 0.95                 | 293                   | 0.17               | 27                       |
| Vgg-16-bn         | 94.99                        | 131                           | 0.95                 | 131                   | 0.2                | 28                       |
| GoogLeNet         | 94.95                        | 136                           | 0.95                 | 136                   | 0.25               | 16                       |
| DenseNet-201      | 94.9                         | 383                           | 0.95                 | 383                   | 0.24               | 11                       |
| Vgg-16            | 94.27                        | 130                           | 0.94                 | 130                   | 0.21               | 20                       |
| EfficientNet-b4   | 94.25                        | 290                           | 0.94                 | 290                   | 0.19               | 30                       |
| MobileFormer-508M | 94.15                        | 129                           | 0.94                 | 129                   | 0.25               | 24                       |
| MobileFormer-294M | 94.11                        | 125                           | 0.94                 | 125                   | 0.26               | 9                        |
| Vgg-11            | 94.08                        | 154                           | 0.94                 | 154                   | 0.22               | 12                       |
| EfficientNet-b0   | 94.03                        | 253                           | 0.94                 | 253                   | 0.2                | 26                       |
| Vgg-13            | 94                           | 90                            | 0.94                 | 90                    | 0.22               | 11                       |
| MobileNetV2       | 93.97                        | 442                           | 0.94                 | 442                   | 0.26               | 17                       |
| Vgg-19            | 93.89                        | 176                           | 0.94                 | 176                   | 0.2                | 17                       |
| EfficientNet-b1   | 93.74                        | 371                           | 0.94                 | 371                   | 0.22               | 31                       |
| EfficientNet-b3   | 93.64                        | 154                           | 0.94                 | 154                   | 0.21               | 31                       |
| MobileFormer-151M | 93.62                        | 134                           | 0.94                 | 134                   | 0.23               | 16                       |
| AlexNet           | 93.62                        | 267                           | 0.94                 | 267                   | 0.22               | 19                       |
| MobileFormer-214M | 93.58                        | 444                           | 0.94                 | 444                   | 0.25               | 22                       |
| EfficientNet-v2-s | 93.41                        | 361                           | 0.93                 | 361                   | 0.22               | 28                       |
| CCT-14-sine       | 93.3                         | 120                           | 0.93                 | 120                   | 0.24               | 33                       |
| MobileNetV1       | 93.27                        | 154                           | 0.93                 | 154                   | 0.3                | 460                      |
| MobileFormer-96M  | 93.21                        | 89                            | 0.93                 | 89                    | 0.3                | 26                       |
| CCT-7-sine        | 93.12                        | 170                           | 0.93                 | 170                   | 0.27               | 30                       |
| ResNet18          | 93.05                        | 170                           | 0.93                 | 170                   | 0.29               | 443                      |
| EfficientNet-b2   | 93                           | 257                           | 0.93                 | 257                   | 0.27               | 26                       |
| RegNet-y-16gf     | 93                           | 45                            | 0.93                 | 45                    | 0.31               | 11                       |



## Computational Results on COVID-19 Dataset



## Acknowledgement