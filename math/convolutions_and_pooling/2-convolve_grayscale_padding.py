#!/usr/bin/env python3
"""
2-convolve_grayscale_padding.py
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom zero padding.

    Parameters:
    - images (numpy.ndarray): Array of shape (m, h, w) containing multiple
    grayscale images
        m = number of images
        h = height in pixels of each image
        w = width in pixels of each image
    - kernel (numpy.ndarray): Array of shape (kh, kw) containing the
    convolution kernel
        kh = height of the kernel
        kw = width of the kernel
    - padding (tuple): Tuple of (ph, pw) specifying padding for height and
    width respectively
        ph = padding for height (top and bottom)
        pw = padding for width (left and right)

    Returns:
    - numpy.ndarray: Array containing the convolved images with shape
    (m, output_height, output_width)
      where:
        output_height = h + 2*ph - kh + 1
        output_width = w + 2*pw - kw + 1
    """
    m = images.shape[0]
    h = images.shape[1]
    ph = padding[0]
    kh = kernel.shape[0]
    w = images.shape[2]
    pw = padding[1]
    kw = kernel.shape[1]

    # Aplicar padding con ceros alrededor de las imágenes
    pad_width = ((0, 0), (ph, ph), (pw, pw))
    new_image = np.pad(
        images, pad_width=pad_width, mode="constant", constant_values=0)

    # Calcular tamaño de la imagen resultante después de la convolucion
    h_output = h + (2 * ph) - kh + 1
    w_output = w + (2 * pw) - kw + 1

    # creo la matriz para guardar las imagees
    matriz = np.zeros((m, h_output, w_output))

    for i in range(h_output):
        for j in range(w_output):
            patch = new_image[:, i:i + kh, j:j+kw]
            matriz[:, i, j] = np.sum(patch * kernel, axis=(1, 2))

    return matriz
