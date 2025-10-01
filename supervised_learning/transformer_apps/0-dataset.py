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
