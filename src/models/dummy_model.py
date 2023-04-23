from typing import List, Dict
import tensorflow as tf
import numpy as np
from .basic_model import BasicModel

rng = np.random.default_rng(12345)

class DummyModel(BasicModel):
    def __init__(self, encoder: tf.keras.layers.StringLookup, vocab_size=1024, embedding_size=100):
        super(BasicModel, self).__init__()
        self.encoder = encoder
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.pipeline = tf.keras.Sequential([
            tf.keras.layers.TextVectorization(
                max_tokens=vocab_size,
            ),
            tf.keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=embedding_size,
                mask_zero=True,
            ),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(1)),
            tf.keras.layers.Dense(self.encoder.vocabulary_size(), activation='sigmoid', input_shape=(6000,)),
        ])

    def call(self, inputs, training=None, mask=None):
        #return [1.0 if rng.random() < 1.0/36 else 0.0 for _ in range(self.encoder.vocabulary_size())]
        print(self.pipeline(inputs))
        return self.pipeline(inputs)

    def save_model(self, path: str) -> None:
        path += "\\dummy_model"
        self.pipeline.save(path)
        with open(f"{path}\\categories.txt", 'w') as file:
            file.write(",".join(self.encoder.get_vocabulary()))
