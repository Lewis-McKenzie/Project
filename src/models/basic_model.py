from typing import List
import tensorflow as tf
import numpy as np

class BasicModel(tf.keras.Model):
    def __init__(self, encoder: tf.keras.layers.StringLookup, embedding_matrix: np.ndarray):
        super(BasicModel, self).__init__()
        self.encoder = encoder
        if embedding_matrix is None:
            return
        self.embedding_matrix = embedding_matrix
        self.pipeline = tf.keras.Sequential([
            tf.keras.layers.TextVectorization(
                max_tokens=embedding_matrix.shape[0],
            ),
            tf.keras.layers.Embedding(
                input_dim=embedding_matrix.shape[0],
                output_dim=embedding_matrix.shape[1],
                embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                mask_zero=True,
                #trainable=False,
            ),           
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100)),
            tf.keras.layers.Dense(512, input_shape=(6000,), activation='relu'),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.encoder.vocabulary_size(), activation='sigmoid'),
        ])

    def call(self, inputs, training=None, mask=None):
        return self.pipeline(inputs)
        #return np.where(self.pipeline(inputs) > 0.5, 1.0, 0)

    def adapt_encoder(self, vocab: List[str]) -> None:
        with tf.device("/CPU:0"):
            self.pipeline.layers[0].adapt(vocab)

    def get_vocab(self) -> List[str]:
        return self.pipeline.layers[0].get_vocabulary()

    def invert_all(self, encoded_labels, alpha: float) -> List[List[str]]:
        return [self.invert_multi_hot(encoding, alpha) for encoding in encoded_labels]

    def invert_multi_hot(self, encoded_labels, alpha: float) -> List[str]:
        hot_indices = np.argwhere(encoded_labels >= alpha)[..., 0]
        return np.take(self.encoder.get_vocabulary(), hot_indices)

    def get_config(self):
        return {"encoder": self.encoder, "embedding_matrix": self.embedding_matrix, "invert_all": self.invert_all, "invert_multi_hot": self.invert_multi_hot}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
