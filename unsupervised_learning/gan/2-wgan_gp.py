import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_GP(keras.Model):
    """
    Wasserstein GAN with Gradient Penalty (WGAN-GP).

    Implements a WGAN with gradient penalty to enforce the Lipschitz constraint
    instead of weight clipping.

    Attributes
    ----------
    generator : keras.Model
        Neural network that generates fake samples.
    discriminator : keras.Model
        Neural network that scores samples (critic).
    latent_generator : callable
        Function generating latent vectors (noise).
    real_examples : tf.Tensor
        Real training samples.
    batch_size : int
        Number of samples per batch.
    disc_iter : int
        Number of discriminator updates per generator update.
    learning_rate : float
        Learning rate for both optimizers.
    beta_1 : float
        Beta_1 for Adam optimizer.
    beta_2 : float
        Beta_2 for Adam optimizer.
    lambda_gp : float
        Weight for the gradient penalty term.
    """

    def __init__(
        self, generator, discriminator, latent_generator,
        real_examples, batch_size=200, disc_iter=2,
        learning_rate=0.005, lambda_gp=10
    ):
        """
        Initialize the WGAN-GP with generator, discriminator,
        losses, optimizers, and parameters for gradient penalty.

        Parameters
        ----------
        generator : keras.Model
            The generator network.
        discriminator : keras.Model
            The discriminator (critic) network.
        latent_generator : callable
            Function generating latent vectors.
        real_examples : tf.Tensor
            Tensor of real samples.
        batch_size : int, optional
            Training batch size (default 200).
        disc_iter : int, optional
            Number of discriminator iterations per generator update (default 2).
        learning_rate : float, optional
            Learning rate for both optimizers (default 0.005).
        lambda_gp : float, optional
            Gradient penalty weight (default 10).
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
        self.discriminator.loss = lambda x, y: tf.reduce_mean(y) - tf.reduce_mean(x)
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta_1,
            beta_2=self.beta_2
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
            Number of fake samples (default self.batch_size).
        training : bool, optional
            Whether the generator is in training mode.

        Returns
        -------
        tf.Tensor
            Generated fake samples.
        """
        if not size:
            size = self.batch_size
        return self.generator(
            self.latent_generator(size), training=training
        )

    def get_real_sample(self, size=None):
        """
        Sample a batch of real examples from the dataset.

        Parameters
        ----------
        size : int, optional
            Number of samples (default self.batch_size).

        Returns
        -------
        tf.Tensor
            Randomly selected real samples.
        """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def get_interpolated_sample(self, real_sample, fake_sample):
        """
        Generate interpolated samples between real and fake examples.

        Parameters
        ----------
        real_sample : tf.Tensor
            Batch of real samples.
        fake_sample : tf.Tensor
            Batch of fake samples.

        Returns
        -------
        tf.Tensor
            Interpolated samples.
        """
        u = tf.random.uniform(self.scal_shape)
        v = tf.ones(self.scal_shape) - u
        return u * real_sample + v * fake_sample

    def gradient_penalty(self, interpolated_sample):
        """
        Compute the gradient penalty for a batch of interpolated samples.

        Parameters
        ----------
        interpolated_sample : tf.Tensor
            Interpolated samples between real and fake examples.

        Returns
        -------
        tf.Tensor
            Gradient penalty scalar.
        """
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

        Parameters
        ----------
        useless_argument : any
            Required by keras.Model.fit but unused.

        Returns
        -------
        dict
            Dictionary with discriminator loss, generator loss, and gradient penalty:
            {"discr_loss": discr_loss, "gen_loss": gen_loss, "gp": gp}
        """
        # Entrenar Discriminador
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                # obtener muestras reales y falsas
                real_sample = self.get_real_sample()
                fake_sample = self.get_fake_sample(training=True)
                interpolated_sample = self.get_interpolated_sample(real_sample, fake_sample)

                # predicciones
                real_pred = self.discriminator(real_sample, training=True)
                fake_pred = self.discriminator(fake_sample, training=True)

                # pérdida tradicional del discriminador
                discr_loss = self.discriminator.loss(fake_pred, real_pred)

                # gradient penalty
                gp = self.gradient_penalty(interpolated_sample)

                # pérdida total
                new_discr_loss = discr_loss + self.lambda_gp * gp

            # aplicar gradientes
            grads = tape.gradient(new_discr_loss, self.discriminator.trainable_variables)
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
        grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables)
        )

        # devolver métricas
        return {"discr_loss": discr_loss, "gen_loss": gen_loss, "gp": gp}
