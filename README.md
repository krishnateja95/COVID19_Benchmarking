# COVID19_Benchmarking

This repository is the official implementation of "Benchmarking and Evaluation of Convolutional Neural Networks and Vision Transformers on COVID-19 Dataset" 


We implemented CNN and ViT models in such a way a single model file contains the entire description of model without having to import any extra module (except torch and torchvision). For example, efficientnet_b0.py file contains the full code for Efficientnet_b0 model. One can directly use the file for any Computer vision application.  


We referred to several open source repositories for [CNN](https://github.com/pytorch/vision/tree/main/torchvision/models) and [ViT](https://github.com/lucidrains/vit-pytorch) models 

## Results


| Model | Accuracy | F1-score | MACs | FLOPs | CPU Latency | GPU Latency | No. of Params | 
|:------|:---------|:--------:|-----:|-------|:------------|:-----------:|---------------|
| ALexNet |  |  |  |  |  |  |  |
| Twins  |  |  |  |  |  |  |  |
| ViT-16 |  |  |  |  |  |  |  |
| Swin |  |  |  |  |  |  |  |
| T2T |  |  |  |  |  |  |  |
| MobileViT |  |  |  |  |  |  |  |
| MobileNet |  |  |  |  |  |  |  |
| MobileNetV2 |  |  |  |  |  |  |  |
| MobileNetV3 |  |  |  |  |  |  |  |
| EfficientNet |  |  |  |  |  |  |  |
| EfficientNetV2 |  |  |  |  |  |  |  |
