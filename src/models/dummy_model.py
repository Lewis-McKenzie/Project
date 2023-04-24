from typing import List, Dict
import tensorflow as tf
import numpy as np
from .basic_model import BasicModel

class DummyModel(BasicModel):
    def __init__(self, encoder: tf.keras.layers.StringLookup):
        super(BasicModel, self).__init__()
        self.encoder = encoder
        self.pipeline = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: tf.where(tf.random.uniform((tf.shape(x)[0], 37)) < 1/37, 1.0, 0.0)),
        ])

    def call(self, inputs, training=None, mask=None):
        return self.pipeline(inputs)

    @classmethod
    def model_name(cls) -> str:
        return "dummy_model"