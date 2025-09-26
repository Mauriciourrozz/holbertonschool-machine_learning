#!/usr/bin/env python3
"""
2-rnn_decoder.py
"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    RNNDecoder is a custom Keras Layer that decodes sequences for
    machine translation. It uses an embedding layer, GRU, and
    Bahdanau-style self-attention to produce the output words.
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor.

        Args:
            vocab (int): Size of the output vocabulary.
            embedding (int): Dimensionality of the embedding vectors.
            units (int): Number of hidden units in the GRU cell.
            batch (int): Batch size.

        Public Attributes:
            embedding (tf.keras.layers.Embedding): Embedding layer that
                converts words into embedding vectors.
            gru (tf.keras.layers.GRU): GRU layer with `units` hidden units,
                returning both sequences and the last hidden state.
            F (tf.keras.layers.Dense): Dense layer with `vocab` units
                to predict the next word.
            attention (SelfAttention): SelfAttention layer to compute
                context vectors from encoder outputs.
        """
        super(RNNDecoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units, return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """
        Performs the forward pass of the decoder.

        Args:
            x (tf.Tensor): Tensor of shape (batch, 1), containing
                the previous word as an index in the target vocabulary.
            s_prev (tf.Tensor): Tensor of shape (batch, units),
                containing the previous decoder hidden state.
            hidden_states (tf.Tensor): Tensor of shape (batch,
            input_seq_len, units), containing the outputs of the encoder.

        Returns:
            y (tf.Tensor): Tensor of shape (batch, vocab), containing
                the next word probabilities in the target vocabulary.
            s (tf.Tensor): Tensor of shape (batch, units), containing
                the new decoder hidden state.
        """
        # Inicializamos la capa de atención según el
        # tamaño del estado oculto
        attention_layer = SelfAttention(s_prev.shape[1])

        # Calculamos el vector de contexto a partir del estado
        # oculto previo y las salidas del encoder
        context_vector, _ = attention_layer(s_prev, hidden_states)

        # Obtenemos el embedding de la palabra previa del target
        x_embedded = self.embedding(x)

        # Concatenamos el vector de contexto y el embedding a lo
        # largo del último eje
        decoder_input = tf.concat([tf.expand_dims(context_vector, 1),
                                   x_embedded], axis=-1)

        # Pasamos la entrada concatenada por la GRU
        gru_outputs, new_state = self.gru(decoder_input)

        # Aplanamos las salidas de la GRU para pasarlas a la capa Dense
        flattened_outputs = tf.reshape(gru_outputs, (gru_outputs.shape[0],
                                                     gru_outputs.shape[2]))

        # Calculamos las predicciones de la siguiente palabra y retornamos
        y_pred = self.F(flattened_outputs)

        return y_pred, new_state
