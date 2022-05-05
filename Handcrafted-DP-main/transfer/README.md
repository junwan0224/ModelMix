This folder contains the following files:

1. opacus: the opacus library, the current version is 0.9. Would be be best to update it to the latest version if available.

2. transfer_cifar_mix.py: this is the default function we should call if you want to use randommix and opacus at the same time

3. transfer_cifar.py: this version does not use randommix, it only uses the default opacus

4. extract_cifar_100: extract the mean and variance of the feature of a preloaded model (stored in model_best_pth.tar file).
This function should be called to fit the format of the training sample before training.

5. trains_util_mix.py: include the details of how we randommix during the training process.

6. trains_util.py: include how training happens for transfer_cifar.py
