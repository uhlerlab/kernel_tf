# Transfer Learning with Kernels
Code for Transfer Learning with Kernels 

## Directory Descriptions 

1. cntk_imagenet, ntk_imagenet, laplace_imagenet: 
    1. These directories contain files for training the cntk, ntk, and laplace kernel on ImageNet32.  Users will need to download ImageNet32 and provide the path to this dataset in dataset.py.  
    2. cntk_imagenet also contains code for transferring trained kernels to other datasets (e.g. projected_*.py and scaling_law_*.py). We considered CIFAR10, Oxford Flower 102, and SVHN in this paper.  Users will need to download these datasets and provide appropriate paths to these in dataset.py. 
2. cnn_imagenet contains code for training convolutional neural nets on ImageNet32 and transferring trained models to other image datasets.  
3. cifarc_analysis contains code for training the cntk, ntk, and laplace kernel on CIFAR10 (*_main.py) and then transferring these models to CIFAR10-C (scaling_*.py).
4. cnn_cifarc contains code for training convolutional neural nets on CIFAR10 and then transferring to CIFARC.

## Dependencies
1.  All dependencies are provided in the kernel_tf_env.yml file.  Key libraries used include neural_tangents, jax and pytorch.   
