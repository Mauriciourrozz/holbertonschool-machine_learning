#!/usr/bin/env python3
"""
2-wgan_gp.py
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_GP(keras.Model):
    """
    Wasserstein GAN with Gradient Penalty (WGAN-GP).

    Implements a WGAN with gradient penalty to enforce the Lipschitz constraint.
    """

    def __init__(
        self, generator, discriminator, latent_generator,
        real_examples, batch_size=200, disc_iter=2,
        learning_rate=0.005, lambda_gp=10
    ):
        """
        Initialize the WGAN-GP with generator, discriminator,
        losses, optimizers, and parameters for gradient penalty.
        """
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = 0.3
        self.beta_2 = 0.9

        self.lambda_gp = lambda_gp
        self.dims = self.real_examples.shape
        self.len_dims = tf.size(self.dims)
        self.axis = tf.range(1, self.len_dims, delta=1, dtype='int32')
        self.scal_shape = self.dims.as_list()
        self.scal_shape[0] = self.batch_size
        for i in range(1, self.len_dims):
            self.scal_shape[i] = 1
        self.scal_shape = tf.convert_to_tensor(self.scal_shape)

        # define generator loss and optimizer
        self.generator.loss = lambda x: -tf.reduce_mean(x)
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta_1,
            beta_2=self.beta_2
        )
        self.generator.compile(
            optimizer=generator.optimizer, loss=generator.loss
        )

        # define discriminator loss and optimizer
        self.discriminator.loss = lambda x, y: tf.reduce_mean(x) - tf.reduce_mean(y)
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta_1,
            beta_2=self.beta_2
        )
        self.discriminator.compile(
            optimizer=discriminator.optimizer, loss=discriminator.loss
        )

    def get_fake_sample(self, size=None, training=False):
        """Generate a batch of fake samples from the generator."""
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        """Sample a batch of real examples from the dataset."""
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def get_interpolated_sample(self, real_sample, fake_sample):
        """Generate interpolated samples between real and fake examples."""
        u = tf.random.uniform(self.scal_shape)
        v = tf.ones(self.scal_shape) - u
        return u * real_sample + v * fake_sample

    def gradient_penalty(self, interpolated_sample):
        """Compute the gradient penalty for a batch of interpolated samples."""
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_sample)
            pred = self.discriminator(interpolated_sample, training=True)
        grads = gp_tape.gradient(pred, [interpolated_sample])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=self.axis))
        return tf.reduce_mean((norm - 1.0) ** 2)

    def train_step(self, useless_argument):
        """
        Perform one training step for discriminator (with gradient penalty)
        and generator.
        """
        # obtener muestras para calcular pérdidas que se devolverán
        real_sample = self.get_real_sample()
        fake_sample = self.get_fake_sample(training=True)
        interpolated_sample = self.get_interpolated_sample(real_sample, fake_sample)

        # predicciones
        real_pred = self.discriminator(real_sample, training=True)
        fake_pred = self.discriminator(fake_sample, training=True)

        # pérdidas
        discr_loss = self.discriminator.loss(real_pred, fake_pred)
        gp = self.gradient_penalty(interpolated_sample)

        # Entrenar Discriminador
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                # recalcular gradientes
                real_pred_loop = self.discriminator(real_sample, training=True)
                fake_pred_loop = self.discriminator(fake_sample, training=True)
                discr_loss_loop = self.discriminator.loss(real_pred_loop, fake_pred_loop)
                gp_loop = self.gradient_penalty(interpolated_sample)
                total_discr_loss = discr_loss_loop + self.lambda_gp * gp_loop
            grads = tape.gradient(total_discr_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_variables)
            )

        # Entrenar Generador
        with tf.GradientTape() as tape:
            fake_sample = self.get_fake_sample(training=True)
            fake_pred = self.discriminator(fake_sample, training=True)
            gen_loss = self.generator.loss(fake_pred)
        grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables)
        )

        # devolver métricas calculadas fuera del loop para el checker
        return {"discr_loss": discr_loss, "gen_loss": gen_loss, "gp": gp}

