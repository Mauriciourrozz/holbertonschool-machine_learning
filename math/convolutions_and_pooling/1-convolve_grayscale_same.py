#!/usr/bin/env python3
"""
1-convolve_grayscale_same.py
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images using a given kernel.

    Parameters:
    - images: numpy.ndarray of shape (m, alto_imagen, ancho_imagen)
    - kernel: numpy.ndarray of shape (alto_kernel, ancho_kernel)

    Returns:
    - numpy.ndarray of shape (m, alto_imagen, ancho_imagen)
    """
    m, alto_imagen, ancho_imagen = images.shape
    alto_kernel, ancho_kernel = kernel.shape

    pad_alto = alto_kernel // 2
    pad_ancho = ancho_kernel // 2

    padded = np.pad(
        images,
        ((0, 0), (pad_alto, pad_alto), (pad_ancho, pad_ancho)),
        mode='constant'
    )

    matriz = np.zeros((m, alto_imagen, ancho_imagen))

    for i in range(alto_imagen):
        for j in range(ancho_imagen):
            patch = padded[:, i:i+alto_kernel, j:j+ancho_kernel]
            matriz[:, i, j] = np.sum(patch * kernel, axis=(1, 2))

    return matriz
