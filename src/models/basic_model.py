from typing import List
import tensorflow as tf

VOCAB_SIZE = 1024

class BasicModel(tf.keras.Model):
    def __init__(self):
        super(BasicModel, self).__init__()
        self.pipeline = tf.keras.Sequential([
            tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE),
            tf.keras.layers.Embedding(
                input_dim=VOCAB_SIZE,
                output_dim=64,
                # Use masking to handle the variable sequence lengths
                mask_zero=True),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(512, input_shape=(6000,), activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(12, activation='softmax'),
        ])

    def call(self, inputs, training=None, mask=None):
        return self.pipeline(inputs)

    def adapt_encoder(self, vocab: List[str]) -> None:
        self.pipeline.layers[0].adapt(vocab)
