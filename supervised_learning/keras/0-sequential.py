#!/usr/bin/env python3
import tensorflow.keras as k

def build_model(nx, layers, activations, lambtha, keep_prob):
    model = k.Sequential()

    # Primera capa
    model.add(
        k.layers.Dense(
            units=layers[0],
            activation=activations[0],
            input_shape=(nx,),
            kernel_regularizer=k.regularizers.l2(lambtha)
        )
    )
    if len(layers) > 1:
        model.add(k.layers.Dropout(1 - keep_prob))

    # se agrega el resto de las capas, de la 2da en adelante
    for i in range(1, len(layers)):
        model.add(
            k.layers.Dense(
                units=layers[i],
                activation=activations[i],
                kernel_regularizer=k.regularizers.l2(lambtha)
            )
        )

        if i != len(layers) - 1:
                model.add(k.layers.Dropout(1 - keep_prob))


    return model
