#!/usr/bin/env python3
import tensorflow.keras as k

def build_model(nx, layers, activations, lambtha, keep_prob):

    # se cargan los datos de entrada al modelo y se guardan en inputs
    inputs =  k.layers.Input(shape=(nx,))

    # se crea una capa densa y al poner (inputs) se conecta esa capa con la entrada
    x = k.layers.Dense(units=layers[0], activation=activations[0], kernel_regularizer=k.regularizers.l2(lambtha))(inputs)

    # se recorren las capaz intermedias, la primera ya fue creada antes
    # y la ultima se hace despues
    for i in range(1, len(layers) - 1):

        # se crea una nueva capa en cada iteracion y se conecta con la anterior
        # que fue x
        x = k.layers.Dense(units=layers[i], activation=activations[i], kernel_regularizer=k.regularizers.l2(lambtha))(x)

        # dropout apaga neuronas al azar antes de seguir para que
        # la red no se apoye demasiado en ciertas neuronas
        x = k.layers.Dropout(1 - keep_prob)(x)

    # ultima capa, procesa los ultimos valores y se conecta a la
    # ultima capa intermedia
    output = x
    # se esta  creando el modelo completo, conectando todas las capas
    model = k.Model(inputs=inputs, outputs=output)

    return model