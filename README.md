# Implementation of ModelMix for Private training

See our paper "Differentially Private Deep Learning with ModelMix" for more details.

# Environment
This code is tested on Linux system with CUDA version 11.3

To run the source code, please first install the following packages:

    python>=3.6
    numpy>=1.15
    torch>=1.11
    torchvision>=0.12
    kymatio

# Folders

The three folders correspond to three different codes / projects. Each folder contains another readme file that describes how the code runs.

cifar10main: ModelMix implementation for cifar10.
Handcrafted-DP-main: ModelMix combined with Tramèr and Boneh's feature extraction method.
Gradient-Embedding-Perturbation-master: ModelMix combined with Gradient Embeding Perturbation.

