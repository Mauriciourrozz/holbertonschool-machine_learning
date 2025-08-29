#!/usr/bin/env python3
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class SimpleGAN(keras.Model):
    """
    Simple implementation of a Generative Adversarial Network (GAN).

    This class manages the generator and discriminator models,
    their training process, and provides utilities for sampling
    real and fake data.

    Attributes
    ----------
    generator : keras.Model
        Neural network that generates fake samples from latent space.
    discriminator : keras.Model
        Neural network that distinguishes real samples from fake ones.
    latent_generator : callable
        Function that generates latent vectors (noise).
    real_examples : tf.Tensor
        Dataset of real samples to be used for training.
    batch_size : int
        Number of samples per training batch.
    disc_iter : int
        Number of discriminator training iterations per step.
    learning_rate : float
        Learning rate for both generator and discriminator optimizers.
    beta_1 : float
        Beta_1 parameter for Adam optimizer.
    beta_2 : float
        Beta_2 parameter for Adam optimizer.
    """

    def __init__(
        self, generator, discriminator,
        latent_generator, real_examples,
        batch_size=200, disc_iter=2, learning_rate=0.005
    ):
        """
        Initialize the GAN model with generator, discriminator,
        optimizers and loss functions.

        Parameters
        ----------
        generator : keras.Model
            The generator network.
        discriminator : keras.Model
            The discriminator network.
        latent_generator : callable
            Function to generate latent vectors.
        real_examples : tf.Tensor
            Tensor containing real training samples.
        batch_size : int, optional
            Training batch size (default is 200).
        disc_iter : int, optional
            Number of discriminator iterations per training step
            (default is 2).
        learning_rate : float, optional
            Learning rate for both optimizers (default is 0.005).
        """
        super().__init__()  # run the __init__ of keras.Model first
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = 0.5  # standard value, but can be changed
        self.beta_2 = 0.9  # standard value, but can be changed

        # define the generator loss and optimizer:
        self.generator.loss = (
            lambda x: tf.keras.losses.MeanSquaredError()(
                x, tf.ones(x.shape)
            )
        )
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
        )
        self.generator.compile(
            optimizer=generator.optimizer, loss=generator.loss
        )

        # define the discriminator loss and optimizer:
        self.discriminator.loss = (
            lambda x, y: tf.keras.losses.MeanSquaredError()(
                x, tf.ones(x.shape)
            )
            + tf.keras.losses.MeanSquaredError()(
                y, -1 * tf.ones(y.shape)
            )
        )
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
        )
        self.discriminator.compile(
            optimizer=discriminator.optimizer, loss=discriminator.loss
        )

    def get_fake_sample(self, size=None, training=False):
        """
        Generate a batch of fake samples from the generator.

        Parameters
        ----------
        size : int, optional
            Number of fake samples to generate. If None,
            self.batch_size is used.
        training : bool, optional
            Whether to run the generator in training mode
            (default is False).

        Returns
        -------
        tf.Tensor
            Tensor of generated fake samples.
        """
        if not size:
            size = self.batch_size
        return self.generator(
            self.latent_generator(size), training=training
        )

    def get_real_sample(self, size=None):
        """
        Sample a random batch of real examples from the dataset.

        Parameters
        ----------
        size : int, optional
            Number of real samples to return. If None,
            self.batch_size is used.

        Returns
        -------
        tf.Tensor
            Tensor containing randomly selected real samples.
        """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def train_step(self, useless_argument):
        """
        Perform one training step for both discriminator and generator.

        Parameters
        ----------
        useless_argument : any
            Argument required by keras.Model.fit but not used
            in this implementation.

        Returns
        -------
        dict
            Dictionary with discriminator and generator losses:
            {"discr_loss": discr_loss, "gen_loss": gen_loss}
        """
        # --- Entrenar Discriminador ---
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                # muestras reales y falsas
                real_sample = self.get_real_sample()
                fake_sample = self.get_fake_sample(training=True)

                # predicciones
                real_pred = self.discriminator(real_sample, training=True)
                fake_pred = self.discriminator(fake_sample, training=True)

                # pérdida del discriminador
                discr_loss = self.discriminator.loss(real_pred, fake_pred)

            # aplicar gradientes
            grads = tape.gradient(
                discr_loss, self.discriminator.trainable_variables
            )
            self.discriminator.optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_variables)
            )

        # Entrenar Generador
        with tf.GradientTape() as tape:
            fake_sample = self.get_fake_sample(training=True)
            fake_pred = self.discriminator(fake_sample, training=True)

            # pérdida del generador
            gen_loss = self.generator.loss(fake_pred)

        # aplicar gradientes
        grads = tape.gradient(
            gen_loss, self.generator.trainable_variables
        )
        self.generator.optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables)
        )

        # devolver métricas
        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
