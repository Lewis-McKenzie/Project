import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils import Loader, Preprocessor
from models import BasicModel
from argumentation import Argument
import tensorflow as tf
from train_sandbox import DIR, EMB_PATH, MODEL_WEIGHTS

def main() -> None:    
    df = Loader.load(DIR)
    processor = Preprocessor(df)
    train_dataset, validation_dataset = processor.make_category_dataset(16)

    embeddings_index = Loader.load_word_embedings(EMB_PATH)

    LR = 0.005
    EPOCHS = 30

    model = BasicModel(processor.category_encoder, len(processor.get_vocab()), name="acd_model")
    model.adapt_encoder(df["text"])
    model.init_embeddings(embeddings_index)
    ALPHA = 0.5
    model.compile(loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                metrics=['binary_accuracy', tf.keras.metrics.Precision(thresholds=ALPHA), tf.keras.metrics.Recall(thresholds=ALPHA), tf.keras.metrics.F1Score(threshold=ALPHA)])

    model.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS, verbose=1)
    model.save_model(MODEL_WEIGHTS)

    INDEX = 0

    predict = model.predict(df["text"])
    print(df["text"][INDEX])
    print(processor.get_category_encoded_labels()[INDEX])
    print(predict[INDEX])
    print(model.invert_multi_hot(processor.get_category_encoded_labels()[INDEX], ALPHA))
    print(model.invert_multi_hot(predict[INDEX], ALPHA))


if __name__ == "__main__":
    main()
