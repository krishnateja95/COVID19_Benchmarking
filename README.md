# ConVision Benchmark: A Contemporary Framework to Benchmark CNN and ViT Models
 
 🧑‍💻 **Authors:** Shreyas Bangalore Vijayakumar, Krishna Teja Chitty-Venkata, Kanishk Arya and Arun K. Somani

🏣 **Affliation:** Iowa State University, Ames, IA

This repository is the official implementation of "[ConVision Benchmark](https://www.mdpi.com/2673-2688/5/3/56)" paper 


We implemented CNN and ViT models in such a way a single model file contains the entire description of model without having to import any extra module (except torch and torchvision). For example, efficientnet_b0.py file contains the full code for Efficientnet_b0 model. One can directly use the file for any Computer vision application.  


We referred to several open source repositories for [CNN](https://github.com/pytorch/vision/tree/main/torchvision/models) and [ViT](https://github.com/lucidrains/vit-pytorch) models 

## Results


| Model-Name | Best-Top-1 | Best-F1-score | Best-Loss | Best-Precision | Best-Recall | Best-FPR | Best-FNR | Best-MCC | MACs | FLOPS | Number-of-Parameters | CPU-latency | GPU-latency  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet18 |  |  |  |  |  |  |  | | |  |  |  |  |


