import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils import Loader, Preprocessor
from models import DummyModel
import tensorflow as tf
from test_sandbox import DIR, MODEL_WEIGHTS, EMB_PATH


def main() -> None:    
    df = Loader.load(DIR)
    processor = Preprocessor(df)
    LR = 0.005

    model = DummyModel(processor.polarity_category_encoder)
    ALPHA = 0.5
    model.compile(loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                metrics=['binary_accuracy', tf.keras.metrics.Precision(thresholds=ALPHA), tf.keras.metrics.Recall(thresholds=ALPHA), tf.keras.metrics.F1Score(threshold=ALPHA)])
    labels = tf.ragged.constant(Preprocessor.polarity_category_values(df))
    encoded_labels = model.encoder(labels).numpy()
    model.evaluate(df["text"], encoded_labels)
    model.save_model(MODEL_WEIGHTS)

    INDEX = 0

    predict = model.predict(df["text"])
    print(df["text"][INDEX])
    print(processor.get_encoded_labels()[INDEX])
    print(predict[INDEX])
    print(model.invert_multi_hot(processor.get_encoded_labels()[INDEX], ALPHA))
    print(model.invert_multi_hot(predict[INDEX], ALPHA))


if __name__ == "__main__":
    main()
