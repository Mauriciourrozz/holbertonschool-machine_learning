#!/usr/bin/env python3
"""
5-train.py
"""
import tensorflow as tf
Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """
    Creates and trains a Transformer model for Portuguese to English
    translation.
    Args:
        N: number of blocks in encoder and decoder.
        dm: dimensionality of the model.
        h: number of attention heads.
        hidden: number of units in the fully connected layers.
        max_len: maximum token length per sequence.
        batch_size: batch size for training.
        epochs: number of training epochs.
    Returns:
        model: trained Transformer model.
    """

    # Inicializar dataset
    dataset = Dataset(batch_size, max_len)
    input_vocab_size = dataset.tokenizer_pt.vocab_size + 2
    target_vocab_size = dataset.tokenizer_en.vocab_size + 2

    # Crear Transformer
    model = Transformer(N, dm, h, hidden,
                        input_vocab=input_vocab_size,
                        target_vocab=target_vocab_size,
                        max_seq_input=max_len,
                        max_seq_target=max_len)

    # Learning rate
    class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, d_model, warmup_steps=4000):
            super().__init__()
            self.d_model = tf.cast(d_model, tf.float32)
            self.warmup_steps = warmup_steps

        def __call__(self, step):
            arg1 = tf.math.rsqrt(tf.cast(step, tf.float32))
            arg2 = tf.cast(step, tf.float32) * (self.warmup_steps ** -1.5)
            return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    learning_rate = CustomSchedule(dm)
    optimizer = tf.keras.optimizers.Adam(learning_rate,
                                         beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)

    # Loss function ignorando tokens de padding
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none'
    )

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    # Función de entrenamiento de un batch
    @tf.function
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_mask, combined_mask, dec_mask = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions = model(inp, tar_inp, True,
                                enc_mask, combined_mask, dec_mask)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return loss, tf.reduce_mean(
            tf.cast(tf.equal(
                tar_real, tf.argmax(
                    predictions, axis=-1,
                    output_type=tf.int32)), tf.float32))

    # Loop de entrenamiento
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0

        for batch, (inp, tar) in enumerate(dataset.data_train, 1):
            batch_loss, batch_acc = train_step(inp, tar)
            total_loss += batch_loss
            total_acc += batch_acc
            num_batches += 1

            if batch % 50 == 0:
                print(f'Epoch {epoch}, batch {batch}: '
                      f'loss {batch_loss:.4f} accuracy {batch_acc:.4f}')

        # Resumen al final de la epoch
        print(f'Epoch {epoch}: loss {total_loss / num_batches:.4f} '
              f'accuracy {total_acc / num_batches:.4f}\n')

    return model
