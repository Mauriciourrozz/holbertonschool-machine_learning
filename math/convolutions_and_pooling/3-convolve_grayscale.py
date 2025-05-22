#!/usr/bin/env python3
"""
3-convolve_grayscale.py
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images.

    Parameters:
    - images (numpy.ndarray): Array of shape (m, h, w) containing multiple
    grayscale images, where m is the number of images, h is the height,
    w is the width.
    - kernel (numpy.ndarray): Array of shape (kh, kw) containing the
    convolution kernel, where kh is the kernel height, kw is the kernel width.
    - padding (str or tuple): Either 'same', 'valid', or a tuple (ph, pw)
    specifying the padding for height and width respectively. 'same'
    applies padding to keep the output size equal to input size.
    'valid' means no padding.
    - stride (tuple): Tuple (sh, sw) specifying the stride for height and
    width.

    Returns:
    - numpy.ndarray: Array containing the convolved images with shape
    (m, new_h, new_w), where new_h and new_w are the dimensions after
    convolution.
    """
    m, h_images, w_images = images.shape
    h_kernel, w_kernel = kernel.shape
    h_stride, w_stride = stride

    if padding == 'same':
        ph = int(np.ceil(((
            h_images - 1) * h_stride + h_kernel - h_images) / 2))
        pw = int(np.ceil(((
            w_images - 1) * w_stride + w_kernel - w_images) / 2))
    if padding == 'valid':
        ph = 0
        pw = 0
    if isinstance(padding, tuple):
        ph = padding[0]
        pw = padding[1]

    pad_width = ((0, 0), (ph, ph), (pw, pw))
    new_image = np.pad(
        images, pad_width=pad_width, mode="constant", constant_values=0)

    h_out = int(((h_images + 2 * ph - h_kernel) / h_stride) + 1)
    w_out = int(((w_images + 2 * pw - w_kernel) / w_stride) + 1)
    output = np.zeros((m, h_out, w_out))

    for i in range(h_out):
        for j in range(w_out):
            start_i = i * h_stride
            end_i = start_i + h_kernel
            start_j = j * w_stride
            end_j = start_j + w_kernel

            image_slice = new_image[:, start_i:end_i, start_j:end_j]

            conv_sum = np.sum(image_slice * kernel, axis=(1, 2))

            output[:, i, j] = conv_sum

    return output
