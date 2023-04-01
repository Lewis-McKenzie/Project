from typing import List
import tensorflow as tf
import numpy as np

VOCAB_SIZE = 1024

class BasicModel(tf.keras.Model):
    def __init__(self, encoder: tf.keras.layers.StringLookup, embedding_matrix: np.ndarray):
        super(BasicModel, self).__init__()
        self.encoder = encoder
        self.pipeline = tf.keras.Sequential([
            tf.keras.layers.TextVectorization(
                max_tokens=embedding_matrix.shape[0],
            ),
            tf.keras.layers.Embedding(
                input_dim=embedding_matrix.shape[0],
                output_dim=embedding_matrix.shape[1],
                embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                mask_zero=True),           
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(512, input_shape=(6000,), activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.encoder.vocabulary_size(), activation='sigmoid'),
        ])

    def call(self, inputs, training=None, mask=None):
        return self.pipeline(inputs)

    def adapt_encoder(self, vocab: List[str]) -> None:
        self.pipeline.layers[0].adapt(vocab)

    def get_vocab(self) -> List[str]:
        return self.pipeline.layers[0].get_vocabulary()

    def invert_all(self, encoded_labels, alpha: float) -> List[List[str]]:
        return [self.invert_multi_hot(encoding, alpha) for encoding in encoded_labels]

    def invert_multi_hot(self, encoded_labels, alpha: float) -> List[str]:
        hot_indices = np.argwhere(encoded_labels >= alpha)[..., 0]
        return np.take(self.encoder.get_vocabulary(), hot_indices)
