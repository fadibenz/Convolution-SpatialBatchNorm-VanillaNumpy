import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    out = x.reshape(x.shape[0], -1).dot(w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    dx = dout.dot(w.T).reshape(x.shape)
    dw = x.reshape(x.shape[0], -1).T.dot(dout)
    db = np.sum(dout, axis=0)
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    dx = (x > 0) * dout
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the mean
    and variance of each feature, and these averages are used to normalize data
    at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7 implementation
    of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D) Normalized X, after scaling and shifting with learnable parameters. 
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #############################################################################
        # Implementation of the training-time forward pass for batch normalization. #
        # I used the minibatch statistics to compute the mean and variance,         #
        # I then uses these statistics to normalize the incoming data,              #
        # and scale and shift the normalized data using gamma and beta.             #                       #
        #                                                                           #
        # All intermediates values needeed for the the backward pass were stored    #
        # in the cache variable.                                                    #
        #############################################################################

        sample_mean = np.mean(x, axis =0) # Result is (d,)
        sample_var = np.var(x, axis = 0, ddof=0)
        running_mean = running_mean * momentum + (1 - momentum) * sample_mean 
        running_var = running_var * momentum + (1 - momentum) * sample_var
        Z = (x - sample_mean) / np.sqrt(sample_var + eps)
        out = gamma * Z + beta   
        cache = (sample_mean, sample_var, Z, x, eps, gamma)  
    elif mode == 'test':

        ##############################################################################
        # Implementation of the test-time forward pass for batch normalization.      #
        # I Used the running mean and variance to normalize the incoming data, then  # 
        # scaled  and shifted the normalized data using gamma and beta.              #
        # I Stored the result in  the out variable.                                  #                       #
        ##############################################################################
        Z = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * Z + beta   
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    sample_mean, sample_var, Z, x, eps, gamma = cache
    N = x.shape[0]
    dx, dgamma, dbeta = None, None, None
    #############################################################################
    # Implementation of the backward pass for batch normalization.              #
    # Results are in the dx, dgamma, and dbeta variables.                       #
    #############################################################################
    dgamma = np.sum( dout * Z, axis=0)
    dbeta = np.sum(dout, axis=0)

    inv_std = 1.0 / np.sqrt(sample_var + eps) 
    dout_sum = np.sum(dout, axis=0) 
    X_minus_mean = x - sample_mean
    dout_X_minus_mean_sum = np.sum(dout * X_minus_mean, axis=0)

    dx = gamma * inv_std * (
        dout - (1.0 / N) * dout_sum - (1.0 / N) * inv_std ** 2 * dout_X_minus_mean_sum * X_minus_mean
    )
    
    return dx, dgamma, dbeta

def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout
        and rescale the outputs to have the same mean as at test time.
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not in
        real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        ###########################################################################
        # Implementation of the training phase forward pass for inverted dropout. #
        ###########################################################################
        mask = (np.random.rand(*x.shape) < p).astype(np.float32)
        out = (1/p) * x * mask
    elif mode == 'test':
        out = x

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)
    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']
    p = dropout_param['p']

    dx = None
    if mode == 'train':
        ###########################################################################
        # Implementation of the training phase backward pass for inverted dropout.#
        ###########################################################################
        dx = 1/p * dout * mask
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None

    #############################################################################
    # Implementation of the convolutional forward pass.                         #
    #############################################################################
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']

    H_new = 1 + (H + 2 * pad - HH) // stride
    W_new = 1 + (W + 2 * pad - WW) // stride
    out = np.zeros((N, F, H_new, W_new))
    x_padded = np.pad(x, ((0,0), (0,0), (pad, pad), (pad, pad)), mode='constant')

    for n in range(0, N):
      for f in range(0, F):
        for i, j in np.ndindex(out.shape[2:]): 
          receptive_field = x_padded[n, :, i * stride : i * stride + HH, j * stride : j * stride + WW]
          out[n][f][i][j] = np.sum( w[f] * receptive_field ) + b[f] 

    cache = (x, w, b, conv_param)
    return out, cache

import numpy as np

def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives, of shape (N, F, H_out, W_out)
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, C, H, W)
    - dw: Gradient with respect to w, of shape (F, C, HH, WW)
    - db: Gradient with respect to b, of shape (F,)
    """
    x, w, b, conv_param = cache
    stride = conv_param['stride']
    pad = conv_param['pad']

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    N, F, H_out, W_out = dout.shape

    x_padded = np.pad(x, 
                      ((0, 0), (0, 0), (pad, pad), (pad, pad)),
                      mode='constant', constant_values=0)
    
    dx_padded = np.zeros_like(x_padded)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    for f in range(F):
        db[f] = np.sum(dout[:, f, :, :])
    
    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    h_end = h_start + HH
                    w_start = j * stride
                    w_end = w_start + WW
                    receptive_field = x_padded[n, :, h_start:h_end, w_start:w_end]                    
                    dout_val = dout[n, f, i, j]
                    dw[f] += dout_val * receptive_field
                    dx_padded[n, :, h_start:h_end, w_start:w_end] += dout_val * w[f]

    if pad != 0:
        dx = dx_padded[:, :, pad:-pad, pad:-pad]
    else:
        dx = dx_padded

    return dx, dw, db



def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    #############################################################################
    # Implementation of the max pooling forward pass                            #
    #############################################################################
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    N, C, H, W = x.shape

    H_new = 1 + (H  - pool_height) // stride
    W_new = 1 + (W  - pool_width) // stride
    out = np.zeros((N, C, H_new, W_new))
    
    for n in range(0, N):
      for c in range(0, C):
        for i, j in np.ndindex(out.shape[2:]): 
          receptive_field = x[n, c, i * stride : i * stride + pool_height, 
                             j * stride : j * stride + pool_width]
          
          out[n, c, i, j] = np.max(receptive_field) 
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.
    
    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    x, pool_param = cache 
    N, C, H, W = x.shape

    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    H_new = 1 + (H  - pool_height) // stride
    W_new = 1 + (W  - pool_width) // stride
    dx = np.zeros_like(x)

    for n in range(N):
      for c in range(C):
        for i in range(H_new):
          for j in range(W_new):
              receptive_field = x[n, c, i * stride : i * stride + pool_height, 
                                  j * stride : j * stride + pool_width]
              max_index = np.unravel_index(np.argmax(receptive_field), receptive_field.shape)
              dx[n, c, i * stride + max_index[0], j * stride + max_index[1]] += dout[n, c, i, j]
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features
    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, {}
    #############################################################################
    # Implementation of the forward pass for spatial batch normalization.       #
    #                                                                           #                       #
    #############################################################################
    N, C, H, W = x.shape
    out = np.zeros_like(x)
    for channel in range(C):
      flattened_array_channel = x[:, channel, : , :].reshape(N, -1)
      out_channel, cache_channel = batchnorm_forward(flattened_array_channel, gamma[channel],
                   beta[channel], bn_param)
      out[:, channel, :, :] = out_channel.reshape(N, H, W)
      cache.update({'channel' + str(channel): cache_channel})
    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass
    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    N, C, H, W = dout.shape
    dx, dgamma, dbeta = np.zeros_like(dout), np.zeros((C)), np.zeros((C))
    #############################################################################
    # Implementation of the backward pass for spatial batch normalization.      #
    #############################################################################
    for channel in range(C):
      flattened_dout_channel = dout[:, channel, : , :].reshape(N, -1)
      dx_channel, dgamma_channel, dbeta_channel = batchnorm_backward(flattened_dout_channel, 
                                            cache[f'channel{str(channel)}'])
      dx[:, channel, :, :] = dx_channel.reshape(N, H, W)
      dgamma[channel] = np.sum(dgamma_channel)
      dbeta[channel] = np.sum(dbeta_channel)
      
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
