import numpy as np
from deeplearning.layer_utils import *

class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, 
                 filter_size=7, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0, dtype=np.float32):
        """
        Initialize a new network.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        C, H, W = input_dim
        self.params['W1'] = np.random.normal(0, weight_scale, 
            (num_filters, C, filter_size, filter_size))
        self.params['b1'] = np.zeros(num_filters)
        
        # Calculate spatial dimensions after pooling
        H_pool = H // 2
        W_pool = W // 2
        flattened_size = num_filters * H_pool * W_pool
        
        self.params['W2'] = np.random.normal(0, weight_scale, 
            (flattened_size, hidden_dim))
        self.params['b2'] = np.zeros(hidden_dim)
        
        self.params['W3'] = np.random.normal(0, weight_scale, 
            (hidden_dim, num_classes))
        self.params['b3'] = np.zeros(num_classes)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        # Forward pass
        out_conv, cache_conv = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        conv_out_shape = out_conv.shape  # Save shape for backward
        out_flat = out_conv.reshape(X.shape[0], -1)
        out_hidden, cache_hidden = affine_relu_forward(out_flat, W2, b2)
        scores, cache_scores = affine_forward(out_hidden, W3, b3)

        if y is None:
            return scores

        # Loss calculation with regularization
        data_loss, dout = softmax_loss(scores, y)
        reg_loss = 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
        loss = data_loss + reg_loss

        # Backward pass
        grads = {}
        dout, grads['W3'], grads['b3'] = affine_backward(dout, cache_scores)
        grads['W3'] += self.reg * W3

        dout, grads['W2'], grads['b2'] = affine_relu_backward(dout, cache_hidden)
        grads['W2'] += self.reg * W2

        dout_reshaped = dout.reshape(conv_out_shape)
        _, grads['W1'], grads['b1'] = conv_relu_pool_backward(dout_reshaped, cache_conv)
        grads['W1'] += self.reg * W1

        return loss, grads