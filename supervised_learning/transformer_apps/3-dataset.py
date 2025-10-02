#!/usr/bin/env python3
"""
3-dataset.py
"""
import tensorflow_datasets as tfds
import tensorflow as tf
import transformers


class Dataset:
    """
    Dataset class for Portuguese to English machine translation.
    """

    def __init__(self, batch_size, max_len):
        """
        Initializes the dataset with batching and filtering.
        Args:
            batch_size: Batch size for training and validation.
            max_len: Maximum allowed length for tokenized sequences.
        """
        # Guardar hiperparámetros
        self.batch_size = batch_size
        self.max_len = max_len

        # Cargar dataset TED HRLR (Portugués ↔ Inglés)
        examples, metadata = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            with_info=True,
            as_supervised=True
        )
        self.data_train = examples['train']
        self.data_valid = examples['validation']

        # Crear los tokenizers
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

        # Tokenizar y preparar dataset de entrenamiento
        self.data_train = self.data_train.map(
            self.tf_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        self.data_train = self.data_train.filter(self.filter_max_length)
        self.data_train = self.data_train.cache()
        self.data_train = self.data_train.shuffle(20000)
        self.data_train = self.data_train.padded_batch(
            self.batch_size, padded_shapes=([None], [None])
        )
        self.data_train = self.data_train.prefetch(tf.data.experimental.AUTOTUNE)

        # Tokenizar y preparar dataset de validación
        self.data_valid = self.data_valid.map(
            self.tf_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        self.data_valid = self.data_valid.filter(self.filter_max_length)
        self.data_valid = self.data_valid.padded_batch(
            self.batch_size, padded_shapes=([None], [None])
        )

    def tokenize_dataset(self, data):
        """
        This method creates sub-word tokenizers for the dataset using
        pre-trained models.

        Args:
            data: tf.data.Dataset containing tuples (pt, en),
                  where `pt` is a Portuguese sentence and `en` is the
                  corresponding English sentence.

        Returns:
            tokenizer_pt: Portuguese tokenizer.
            tokenizer_en: English tokenizer.
        """
        # Preparar listas de oraciones en portugués e inglés
        pt_sentences = []
        en_sentences = []

        # Iterar sobre el dataset y decodificar cada ejemplo
        for pt, en in data:
            pt_sentences.append(pt.numpy().decode('utf-8'))
            en_sentences.append(en.numpy().decode('utf-8'))

        # Crear tokenizadores pre-entrenados de Hugging Face
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased',
            use_fast=True,
            clean_up_tokenization_spaces=True
        )
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased',
            use_fast=True,
            clean_up_tokenization_spaces=True
        )

        # Entrenar los tokenizadores en las oraciones del dataset
        tokenizer_pt = tokenizer_pt.train_new_from_iterator(pt_sentences,
                                                            vocab_size=2**13)
        tokenizer_en = tokenizer_en.train_new_from_iterator(en_sentences,
                                                            vocab_size=2**13)

        # Guardar tokenizadores en atributos de la clase
        self.tokenizer_pt = tokenizer_pt
        self.tokenizer_en = tokenizer_en

        # Retornar los tokenizadores
        return self.tokenizer_pt, self.tokenizer_en

    def encode(self, pt, en):
        """
        This method encodes a Portuguese and English sentence into tokens.
        Args:
            pt: tf.Tensor containing the Portuguese sentence.
            en: tf.Tensor containing the English sentence.
        Returns:
            pt_tokens: list containing the Portuguese tokens.
            en_tokens: list containing the English tokens.
        """
        # Decodificamos los tensores a string
        pt_sentence = pt.numpy().decode('utf-8')
        en_sentence = en.numpy().decode('utf-8')

        # Tokenizamos usando los tokenizers ya entrenados
        pt_tokens = self.tokenizer_pt.encode(pt_sentence,
                                             add_special_tokens=False)
        en_tokens = self.tokenizer_en.encode(en_sentence,
                                             add_special_tokens=False)

        # Agregamos tokens de inicio y fin de oración
        pt_tokens = [
            self.tokenizer_pt.vocab_size] + pt_tokens + [
            self.tokenizer_pt.vocab_size + 1]

        en_tokens = [
            self.tokenizer_en.vocab_size] + en_tokens + [
                self.tokenizer_en.vocab_size + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        TensorFlow wrapper for the encode method.
        Args:
            pt: Portuguese sentence tensor.
            en: English sentence tensor.
        Returns:
            pt_tokens: Portuguese tokens as tf.Tensor.
            en_tokens: English tokens as tf.Tensor.
        """
        # Llamar a encode usando py_function
        pt_tokens, en_tokens = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )

        # Definir shape como vector
        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])

        return pt_tokens, en_tokens

    def filter_max_length(self, pt, en):
        """
        Filters sequences longer than max_len.
        """
        return tf.logical_and(tf.size(pt) <= self.max_len,
                              tf.size(en) <= self.max_len)
