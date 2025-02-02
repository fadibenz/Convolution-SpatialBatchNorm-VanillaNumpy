from deeplearning.layers import *


def affine_relu_forward(x, w, b):
    """
    Convenience layer that performs an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


def affine_relu_bn_forward(x, w, b, gamma, beta, bn_param):
  out, cache = None, None
  a, fc_cache = affine_forward(x, w, b)
  b, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
  out, relu_cache = relu_forward(b)
  cache = (fc_cache, bn_cache, relu_cache)
  return out, cache

def affine_relu_bn_backward(dout, cache):
  fc_cache, bn_cache, relu_cache = cache
  drelu = relu_backward(dout, relu_cache)
  dx_n, dgamma, dbeta = batchnorm_backward(drelu, bn_cache)
  dx, dw, db = affine_backward(dx_n, fc_cache)
  return dx, dw, db, dgamma, dbeta



def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
    Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    return dx, dw, db
