#!/usr/bin/env python3
"""
0-conv_forward.py
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
Performs forward propagation in a convolutional layer.

    Parameters:
    - A_prev: form numpy.ndarray (m, h_prev, w_prev, c_prev)
      Output of the previous layer (m = number of examples).
    - W: form numpy.ndarray (kh, kw, c_prev, c_new)
      Convolution kernels (filters).
    - b: form numpy.ndarray (1, 1, 1, c_new)
      Bias (bias) applied to each filter.
    - activation: activation function to apply after convolution.
    - padding: str, "same" or "valid"
      Type of padding to use:
      - "same": output with the same size as input (padding is applied).
      - "valid": without padding (the output will be smaller).
    - stride: tuple (sh, sw)
      Step (stride) for convolution in height and width.

    Returns:
    - A: numpy.ndarray
      Output of the convolutional layer after applying activation.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        ph = (((h_prev - 1) * sh + kh - h_prev) // 2)
        pw = (((w_prev - 1) * sw + kw - w_prev) // 2)
    else:
        ph, pw = 0, 0

    A_prev_padded = np.pad(
        A_prev,
        pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=0
    )

    h_out = (h_prev + 2 * ph - kh) // sh + 1
    w_out = (w_prev + 2 * pw - kw) // sw + 1

    Z = np.zeros((m, h_out, w_out, c_new))

    for i in range(m):
        for h in range(h_out):
            for w in range(w_out):
                for c in range(c_new):
                    h_start = h * sh
                    h_end = h_start + kh
                    w_start = w * sw
                    w_end = w_start + kw
                    slice_img = A_prev_padded[i,
                                              h_start:h_end, w_start:w_end, :]
                    Z[i, h, w, c] = np.sum(slice_img * W[..., c]) + b[..., c]

    A = activation(Z)

    return A
