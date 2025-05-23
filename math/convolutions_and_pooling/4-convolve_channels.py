#!/usr/bin/env python3
"""
4-convolve_channels.py
"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with multiple channels.

    Parameters:
        images (numpy.ndarray): Array of shape (m, h, w, c) containing multiple
        images
            - m: number of images
            - h: height in pixels
            - w: width in pixels
            - c: number of channels
        kernel (numpy.ndarray): Array of shape (kh, kw, c) containing the
        kernel for the convolution
            - kh: kernel height
            - kw: kernel width
            - c: must match the number of image channels
        padding (tuple or str): Either a tuple of (ph, pw), or 'same', or
        'valid'
            - 'same': performs a same convolution (output has same size
            as input)
            - 'valid': performs a valid convolution (no padding)
            - (ph, pw): custom padding for height and width
        stride (tuple): Tuple of (sh, sw)
            - sh: stride for the height
            - sw: stride for the width

    Returns:
        numpy.ndarray: Convolved images with shape (m, output_h, output_w)
    """
    m, h, w, c = images.shape
    kh, kw, _ = kernel.shape
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

    new_images = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        mode='constant', constant_values=0)

    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            vertical_start = i * sh
            vertical_end = vertical_start + kh
            horizontal_start = j * sw
            horizontal_end = horizontal_start + kw

            image_slice = new_images[:, vertical_start:vertical_end,
                                     horizontal_start:horizontal_end, :]

            output[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2, 3))

    return output
