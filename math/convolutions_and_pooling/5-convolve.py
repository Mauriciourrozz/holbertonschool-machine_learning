#!/usr/bin/env python3
"""
5-convolve.py
"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernels.

    Parameters:
        images (numpy.ndarray): Array of shape (m, h, w, c) containing multiple
        images.
            - m: number of images
            - h: image height in pixels
            - w: image width in pixels
            - c: number of channels per image
        kernels (numpy.ndarray): Array of shape (kh, kw, c, nc) containing the
        convolution kernels.
            - kh: kernel height
            - kw: kernel width
            - c: number of channels (must match the images)
            - nc: number of kernels
        padding (tuple, 'same', or 'valid'): Padding strategy to use.
            - 'same': applies padding to keep output size the same as input
            size
            - 'valid': no padding
            - (ph, pw): tuple of height and width padding
        stride (tuple): Tuple of (sh, sw)
            - sh: stride for height
            - sw: stride for width

    Returns:
        numpy.ndarray: Output array of shape (m, output_h, output_w, nc)
            - Contains the result of convolving each image with all kernels.
    """
    m, h, w, c = images.shape
    kh, kw, _, nc = kernels.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))
    if padding == 'valid':
        ph = 0
        pw = 0
    if isinstance(padding, tuple):
        ph, pw = padding

    out_h = ((h + 2 * ph - kh) // sh) + 1
    out_w = ((w + 2 * pw - kw) // sw) + 1

    padded = np.pad(
        images,
        pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=0
    )

    output = np.zeros((m, out_h, out_w, nc))

    for i in range(out_h):
        for j in range(out_w):
            for k in range(nc):
                hs = i * sh
                ws = j * sw
                image_slice = padded[:, hs:hs+kh, ws:ws+kw, :]
                output[:, i, j, k] = np.sum(image_slice * kernels[:, :, :, k],
                                            axis=(1, 2, 3))

    return output
