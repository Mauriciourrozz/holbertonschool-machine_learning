#!/usr/bin/env python3
"""
0-dataset.py
"""
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
    Dataset class for Portuguese to English machine translation.
    """

    def __init__(self):
        """
        Class constructor.
        Loads training and validation splits from the TED Talks pt->en dataset
        and initializes tokenizers.
        """
        # Cargar dataset de entrenamiento y validación
        datos, _ = tfds.load('ted_hrlr_translate/pt_to_en',
                             as_supervised=True,
                             with_info=True)

        self.data_train = datos['train']
        self.data_valid = datos['validation']

        # Crear tokenizadores usando el método tokenize_dataset
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for the dataset using pre-trained models.

        Args:
            data (tf.data.Dataset): Dataset containing tuples (pt, en)

        Returns:
            tuple: (tokenizer_pt, tokenizer_en)
                - tokenizer_pt: Portuguese tokenizer (BERT pretrained)
                - tokenizer_en: English tokenizer (BERT pretrained)
        """
        # Usamos modelos pre-entrenados de Hugging Face
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased"
        )
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased"
        )

        # Definimos el vocabulario máximo
        tokenizer_pt.model_max_length = 2**13
        tokenizer_en.model_max_length = 2**13

        # Retornamos ambos tokenizadores
        return tokenizer_pt, tokenizer_en
