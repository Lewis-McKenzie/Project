import tensorflow as tf

from utils import Preprocessor
from . import BasicModel, VOCAB_SIZE

class PolarityCategoryModel(BasicModel):
    def __init__(self):
        super(BasicModel, self).__init__()
        self.encoder = Preprocessor.polarity_category_encoder
        self.pipeline = tf.keras.Sequential([
            tf.keras.layers.TextVectorization(
                max_tokens=VOCAB_SIZE,
            ),
            tf.keras.layers.Embedding(
                input_dim=VOCAB_SIZE,
                output_dim=128,
                # Use masking to handle the variable sequence lengths
                mask_zero=True),         
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(512, input_shape=(6000,), activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.encoder.vocab_size(), activation='sigmoid'),
        ])