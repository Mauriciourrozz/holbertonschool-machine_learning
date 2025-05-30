#!/usr/bin/env python3
"""
2-conv_backward.py
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs the backward propagation over a convolutional layer of a neural
    network.

    Parameters:
    - dZ (numpy.ndarray): Partial derivatives with respect to the
    unactivated output of the convolutional layer,
    shape (m, h_new, w_new, c_new)
        m: number of examples
        h_new: height of the output
        w_new: width of the output
        c_new: number of channels in the output
    - A_prev (numpy.ndarray): Output of the previous layer,
                              shape (m, h_prev, w_prev, c_prev)
        h_prev: height of the previous layer
        w_prev: width of the previous layer
        c_prev: number of channels in the previous layer
    - W (numpy.ndarray): Convolutional filters,
                         shape (kh, kw, c_prev, c_new)
        kh: filter height
        kw: filter width
    - b (numpy.ndarray): Biases applied to the convolution,
                         shape (1, 1, 1, c_new)
    - padding (str): "same" or "valid", indicating the type of padding used
    - stride (tuple): Tuple of (sh, sw) containing the strides for the
    convolution
        sh: stride height
        sw: stride width

    Returns:
    - dA_prev (numpy.ndarray): Gradient with respect to the previous layer
    (input),shape (m, h_prev, w_prev, c_prev)
    - dW (numpy.ndarray): Gradient with respect to the filters,
                         shape (kh, kw, c_prev, c_new)
    - db (numpy.ndarray): Gradient with respect to the biases,
                         shape (1, 1, 1, c_new)
    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    if padding == "valid":
        ph = 0
        pw = 0
    if padding == "same":
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2

    A_prev_padded = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                           mode='constant', constant_values=0)
    dA_prev_padded = np.pad(dA_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                            mode='constant', constant_values=0)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for j in range(c_new):
                    h_start = h * sh
                    h_end = h_start + kh
                    w_start = w * sw
                    w_end = w_start + kw

                    patch = A_prev_padded[i, h_start:h_end, w_start:w_end, :]
                    dA_prev_padded[i, h_start:h_end, w_start:w_end, :] += W[
                        :, :, :, j] * dZ[i, h, w, j]
                    dW[:, :, :, j] += patch * dZ[i, h, w, j]
                    db[:, :, :, j] += dZ[i, h, w, j]

    if padding == "same":
        dA_prev = dA_prev_padded[:, ph:-ph, pw:-pw, :]
    else:
        dA_prev = dA_prev_padded

    return dA_prev, dW, db
