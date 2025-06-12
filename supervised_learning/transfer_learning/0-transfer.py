#!/usr/bin/env python3
"""
0-transfer.py

"""

import tensorflow as tf
from sklearn.model_selection import train_test_split


def preprocess_data(X, Y):
    """
    Normalize, resize images and convert tags to one-hot.
    Args:
        X: numpy.ndarray with images (samples, 32, 32, 3)
        And: numpy.ndarray with labels (samples,)
    Returns:
        X_p: preprocessed images (samples, 96, 96, 3)
        Y_p: one-hot tags (samples, 10)
    """
    # Normalizar píxeles a rango entre 0 y 1
    x = X / 255.0
    # Redimensionar imágenes a 96x96 por MobileNetV2)
    x = tf.image.resize(x, (96, 96))
    # Convertir etiquetas a one-hot encoding
    y = tf.keras.utils.to_categorical(Y, 10)
    return x, y


# Cargar dataset CIFAR-10
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()

# Dividir conjunto de entrenamiento en entrenamiento y validación
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train, Y_train, test_size=0.2, random_state=42
)

# Preprocesar los datos
X_train, Y_train = preprocess_data(X_train, Y_train)
X_val, Y_val = preprocess_data(X_val, Y_val)


# Definir el modelo MobileNetV2 base con pesos preentrenados
input_shape = (96, 96, 3)
base_model = tf.keras.applications.MobileNetV2(
    weights="imagenet", input_shape=input_shape, include_top=False
)
# Congelar las capas del modelo base para no entrenarlas
base_model.trainable = False

# Construir modelo completo con nuevas capas superiores
inputs = tf.keras.Input(shape=input_shape)
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0, 3)(x)
outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

# Callbacks para detener temprano si no mejora la validación
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Descongelar algunas capas del modelo base
base_model.trainable = True
for layer in base_model.layers[:-80]:
    layer.trainable = False

# Compilar modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)

# Entrenar el modelo
model.fit(
    X_train,
    Y_train,
    epochs=20,
    batch_size=8,
    validation_data=(X_val, Y_val),
    callbacks=[early_stopping]
)

model.save("cifar10.h5")
