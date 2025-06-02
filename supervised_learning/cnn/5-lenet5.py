#!/usr/bin/env python3
"""
4-lenet5.py
"""
from tensorflow import keras as K


def lenet5(X):
    """
    Builds a modified LeNet-5 architecture using Keras

    Args:
        X: K.Input of shape (m, 28, 28, 1) containing the input images

    Returns:
        A K.Model compiled to use Adam optimization and accuracy metric
    """

    # Inicializador He normal con semilla 0
    he_init = K.initializers.he_normal(seed=0)

    # Primera capa convolucional: 6 filtros 5x5, padding 'same',activación ReLU
    conv1 = K.layers.Conv2D(filters=6,
                            kernel_size=(5, 5),
                            padding='same',
                            activation='relu',
                            kernel_initializer=he_init)(X)

    # Primera capa de agrupamiento máximo: 2x2, stride 2x2
    pool1 = K.layers.MaxPooling2D(pool_size=(2, 2),
                                  strides=(2, 2))(conv1)

    # Segunda capa convolucional:16 filtros 5x5,padding 'valid',activación ReLU
    conv2 = K.layers.Conv2D(filters=16,
                            kernel_size=(5, 5),
                            padding='valid',
                            activation='relu',
                            kernel_initializer=he_init)(pool1)

    # Segunda capa de agrupamiento máximo: 2x2, stride 2x2
    pool2 = K.layers.MaxPooling2D(pool_size=(2, 2),
                                  strides=(2, 2))(conv2)

    # Aplanar los datos para las capas totalmente conectadas
    flat = K.layers.Flatten()(pool2)

    # Capa totalmente conectada con 120 nodos y activación ReLU
    fc1 = K.layers.Dense(units=120,
                         activation='relu',
                         kernel_initializer=he_init)(flat)

    # Capa totalmente conectada con 84 nodos y activación ReLU
    fc2 = K.layers.Dense(units=84,
                         activation='relu',
                         kernel_initializer=he_init)(fc1)

    # Capa de salida softmax con 10 nodos (para clasificación en 10 clases)
    output = K.layers.Dense(units=10,
                            activation='softmax',
                            kernel_initializer=he_init)(fc2)

    # Crear y compilar el modelo
    model = K.Model(inputs=X, outputs=output)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
