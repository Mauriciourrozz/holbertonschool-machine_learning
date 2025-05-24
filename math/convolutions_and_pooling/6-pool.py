#!/usr/bin/env python3
"""
6-pool.py
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images using a specified kernel and stride.

    Parameters:
        images (numpy.ndarray): Array of shape (m, h, w, c) containing multiple
        images.
            - m: number of images
            - h: height of each image
            - w: width of each image
            - c: number of channels per image
        kernel_shape (tuple): (kh, kw) specifying the kernel height and width.
        stride (tuple): (sh, sw) specifying the vertical and horizontal stride.
        mode (str): Type of pooling to perform.
            - 'max': applies max pooling
            - 'avg': applies average pooling

    Returns:
        numpy.ndarray: Array of shape (m, output_h, output_w, c) containing
        the pooled images.
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    out_h = (h - kh) // sh + 1
    out_w = (w - kw) // sw + 1

    output = np.zeros((m, out_h, out_w, c))

    for i in range(out_h):
        for j in range(out_w):

            hs = i * sh
            ws = j * sw

            patch = images[:, hs:hs+kh, ws:ws+kw, :]

            if mode == "max":
                output[:, i, j, :] = np.max(patch, axis=(1, 2))

            if mode == "avg":
                output[:, i, j, :] = np.mean(patch, axis=(1, 2))

    return output
