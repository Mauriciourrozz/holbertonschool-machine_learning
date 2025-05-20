#!/usr/bin/env python3
"""
0-convolve_grayscale_valid.py
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images using a given kernel.

    Parameters:
    - images (numpy.ndarray): Array of shape (m, h, w) containing multiple
      grayscale images, where:
        m: number of images
        h: height of each image
        w: width of each image
    - kernel (numpy.ndarray): Array of shape (kh, kw) representing the kernel
      for the convolution.

    Returns:
    - numpy.ndarray: Array of shape (m, new_h, new_w) containing the convolved
      images, where:
        new_h = h - kh + 1
        new_w = w - kw + 1
    """
    m, alto_imagen, ancho_imagen = images.shape
    alto_kernel, ancho_kernel = kernel.shape
    nuevo_alto = alto_imagen - alto_kernel + 1
    nuevo_ancho = ancho_imagen - ancho_kernel + 1
    matriz = np.zeros((m, nuevo_alto, nuevo_ancho))

    for i in range(nuevo_alto):
        for j in range(nuevo_ancho):
            patch = images[:, i:i+alto_kernel, j:j+ancho_kernel]
            matriz[:, i, j] = np.sum(patch * kernel, axis=(1, 2))

    return matriz
