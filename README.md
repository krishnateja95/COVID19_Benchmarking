# ConVision Benchmark: A Contemporary Framework to Benchmark CNN and ViT Models
 
 🧑‍💻 **Authors:** Shreyas Bangalore Vijayakumar, Krishna Teja Chitty-Venkata, Kanishk Arya and Arun K. Somani

🏣 **Affliation:** Iowa State University, Ames, IA

This repository is the official implementation of ["ConVision Benchmark"](https://www.mdpi.com/2673-2688/5/3/56) paper 

## 📌 About
Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) have shown
remarkable performance in computer vision tasks, including object detection and image recognition.
These models have evolved significantly in architecture, efficiency, and versatility. Concurrently,
deep-learning frameworks have diversified, with versions that often complicate reproducibility and
unified benchmarking. We propose ConVision Benchmark, a comprehensive framework in PyTorch,
to standardize the implementation and evaluation of state-of-the-art CNN and ViT models. This
framework addresses common challenges such as version mismatches and inconsistent validation
metrics. As a proof of concept, we performed an extensive benchmark analysis on a COVID-19
dataset, encompassing nearly 200 CNN and ViT models in which DenseNet-161 and MaxViT-Tiny
achieved exceptional accuracy with a peak performance of around 95%. Although we primarily
used the COVID-19 dataset for image classification, the framework is adaptable to a variety of
datasets, enhancing its applicability across different domains. Our methodology includes rigorous
performance evaluations, highlighting metrics such as accuracy, precision, recall, F1 score, and computational efficiency (FLOPs, MACs, CPU, and GPU latency). The ConVision Benchmark facilitates
a comprehensive understanding of model efficacy, aiding researchers in deploying high-performance
models for diverse applications.


We implemented CNN and ViT models in such a way a single model file contains the entire description of model without having to import any extra module (except torch and torchvision). For example, efficientnet_b0.py file contains the full code for Efficientnet_b0 model. One can directly use the file for any Computer vision application.  


We referred to several open source repositories for [CNN](https://github.com/pytorch/vision/tree/main/torchvision/models) and [ViT](https://github.com/lucidrains/vit-pytorch) models 

## Results


| Model-Name | Best-Top-1 | Best-F1-score | Best-Loss | Best-Precision | Best-Recall | Best-FPR | Best-FNR | Best-MCC | MACs | FLOPS | Number-of-Parameters | CPU-latency | GPU-latency  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet18 |  |  |  |  |  |  |  | | |  |  |  |  |


