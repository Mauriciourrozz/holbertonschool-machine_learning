#!/usr/bin/env python3
"""
1-pool_forward.py
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer.

    Parameters:
    - A_prev: numpy.ndarray (m, h_prev, w_prev, c_prev)
      Output of the previous layer.
    - kernel_shape: tuple (kh, kw)
      Core size for pooling.
    - stride: tuple (sh, sw)
      Movement step for pooling.
    - mode: str
      'max' for max pooling, 'avg' for average pooling.

    Returns:
    - A: numpy.ndarray
      Output after applying pooling.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h_out = (h_prev - kh) // sh + 1
    w_out = (w_prev - kw) // sw + 1

    matriz = np.zeros((m, h_out, w_out, c_prev))

    for i in range(m):
        for h in range(h_out):
            for w in range(w_out):
                for c in range(c_prev):
                    h_start = h * sh
                    h_end = h_start + kh
                    w_start = w * sw
                    w_end = w_start + kw

                    slice_img = A_prev[i, h_start:h_end, w_start:w_end, c]

                    if mode == 'max':
                        matriz[i, h, w, c] = np.max(slice_img)
                    elif mode == 'avg':
                        matriz[i, h, w, c] = np.mean(slice_img)

    return matriz
