#!/usr/bin/env python3
"""
3-pool_backward.py
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs backpropagation over a pooling layer of a neural network.

    Parameters:
    - dA (numpy.ndarray): shape (m, h_new, w_new, c_new), contains the partial
    derivatives with respect to the output of the pooling layer.
    - A_prev (numpy.ndarray): shape (m, h_prev, w_prev, c_prev), contains the
    output of the previous layer.
    - kernel_shape (tuple): (kh, kw), the size of the pooling window.
    - stride (tuple): (sh, sw), the stride for the pooling operation.
    - mode (str): 'max' or 'avg', determines whether to perform max pooling
    or average pooling.

    Returns:
    - dA_prev (numpy.ndarray): shape (m, h_prev, w_prev, c_prev),
    containing the partial derivatives with respect to the input of the pooling
    layer (A_prev).
    """
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    sh, sw = stride
    kh, kw = kernel_shape

    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        for j in range(h_new):
            for x in range(w_new):
                for r in range(c_new):
                    h_start = j * sh
                    h_end = h_start + kh
                    w_start = x * sw
                    w_end = w_start + kw

                    a_slice = A_prev[i, h_start:h_end, w_start:w_end, r]

                    if mode == 'max':
                        mask = (a_slice == np.max(a_slice))
                        dA_prev[i, h_start:h_end,
                                w_start:w_end, r] += mask * dA[i, j, x, r]

                    elif mode == 'avg':
                        da = dA[i, j, x, r]
                        average = da / (kh * kw)
                        dA_prev[i, h_start:h_end, w_start:w_end, r] += np.ones(
                            (kh, kw)) * average

    return dA_prev
