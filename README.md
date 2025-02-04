# Convolutional Networks From Scratch in NumPy

## Overview
This notebook explores the implementation and functionality of convolutional networks from scratch, focusing on forward and backward passes for convolutional layers, max pooling, and spatial batch normalization. The notebook is structured to provide a hands-on understanding of these concepts through practical implementation and testing.

## Acknowledgments
The starter code for this notebook is derived from the CS182 course at UC Berkeley. Special thanks to the course instructors and contributors for providing the foundational code and datasets used in this notebook.

## Setup
The notebook begins with necessary imports and setup, including setting random seeds for reproducibility and configuring plot settings. The CIFAR-10 dataset is loaded and preprocessed for use in the experiments.

### Key Imports
- **NumPy**: For numerical operations.
- **Matplotlib**: For plotting and visualization.
- **Torch**: For setting random seeds.
- **Custom Modules**: Includes classifiers, data utilities, gradient checking, and solver modules from the `deeplearning` package.

## Convolutional Layers
### Forward Pass
The notebook implements a naive forward pass for convolutional layers in the `conv_forward_naive` function. This implementation is tested against expected outputs to ensure correctness.

### Backward Pass
The backward pass for convolutional layers is implemented in the `conv_backward_naive` function. Gradient checking is performed to validate the implementation.

## Max Pooling
### Forward Pass
The forward pass for max pooling is implemented in the `max_pool_forward_naive` function. The implementation is tested to ensure it produces the correct output.

### Backward Pass
The backward pass for max pooling is implemented in the `max_pool_backward_naive` function. Gradient checking is used to verify the correctness of the implementation.

## Convolutional "Sandwich" Layers
The notebook introduces and implements commonly used patterns for convolutional networks, such as `conv_relu_pool_forward` and `conv_relu_pool_backward`. These functions are tested for correctness using gradient checking.

## Three-Layer ConvNet
A three-layer convolutional network is implemented in the `ThreeLayerConvNet` class. The notebook includes sanity checks for loss and gradient checking to ensure the network is implemented correctly.

## Spatial Batch Normalization
### Forward Pass
The forward pass for spatial batch normalization is implemented in the `spatial_batchnorm_forward` function. The implementation is tested to ensure it produces the correct means and variances.

### Backward Pass
The backward pass for spatial batch normalization is implemented in the `spatial_batchnorm_backward` function. Gradient checking is used to validate the implementation.

- Torch
