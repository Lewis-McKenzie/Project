from typing import List, Dict
import tensorflow as tf
import numpy as np

class BasicModel(tf.keras.Model):
    def __init__(self, encoder: tf.keras.layers.StringLookup, vocab_size=1024, embedding_size=100, name="basic_model"):
        super(BasicModel, self).__init__()
        self.encoder = encoder
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.model_name = name
        self.pipeline = tf.keras.Sequential([
            tf.keras.layers.TextVectorization(
                max_tokens=vocab_size,
            ),
            tf.keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=embedding_size,
                mask_zero=True,
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

    def adapt_encoder(self, vocab: List[str]) -> None:
        with tf.device("/CPU:0"):
            self.pipeline.layers[0].adapt(vocab)

    def init_embeddings(self, word_embeddings: Dict[str, List[float]]) -> None:
        embedding_matrix = self.get_embedding_matrx(word_embeddings)
        self.pipeline.layers[1] = tf.keras.layers.Embedding(
            input_dim=embedding_matrix.shape[0],
            output_dim=embedding_matrix.shape[1],
            embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
            mask_zero=True,
        )
    
    def get_embedding_matrx(self, word_embeddings: Dict[str, List[float]]) -> np.ndarray:
        voc = self.get_vocab()
        num_tokens = len(voc) + 2
        embedding_dim = len(list(word_embeddings.values())[0])
        word_index = dict(zip(voc, range(len(voc))))
        embedding_matrix = np.zeros((num_tokens, embedding_dim))
        for word, i in word_index.items():
            embedding_vector = word_embeddings.get(word)
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                # This includes the representation for "padding" and "OOV"
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

    def get_vocab(self) -> List[str]:
        return self.pipeline.layers[0].get_vocabulary()

    def invert_all(self, encoded_labels, alpha: float) -> List[List[str]]:
        return [self.invert_multi_hot(encoding, alpha) for encoding in encoded_labels]

    def invert_multi_hot(self, encoded_labels, alpha: float) -> List[str]:
        hot_indices = np.argwhere(encoded_labels >= alpha)[..., 0]
        return np.take(self.encoder.get_vocabulary(), hot_indices)

    def get_config(self):
        return {"encoder": self.encoder, "vocab_size": self.vocab_size, "embedding_size": self.embedding_size}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def save_model(self, path: str) -> None:
        path += f"\\{self.model_name}"
        self.pipeline.save(path)
        with open(f"{path}\\categories.txt", 'w') as file:
            file.write(",".join(self.encoder.get_vocabulary()))
