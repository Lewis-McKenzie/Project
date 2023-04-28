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
    train_dataset, validation_dataset = processor.make_polarity_dataset(16)

    embeddings_index = Loader.load_word_embedings(EMB_PATH)

    LR = 0.005
    EPOCHS = 5

    model = BasicModel(processor.polarity_encoder, len(processor.get_vocab()), name="polarity_model")
    model.adapt_encoder(list(df["text"]) + processor.categories)
    model.init_embeddings(embeddings_index)
    ALPHA = 0.5
    model.compile(loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                metrics=['accuracy', tf.keras.metrics.Precision(thresholds=ALPHA), tf.keras.metrics.Recall(thresholds=ALPHA), tf.keras.metrics.F1Score(threshold=ALPHA)])
    model.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS, verbose=1)
    model.save_model(MODEL_WEIGHTS)

    INDEX = 0

    x, y = Preprocessor.pair_text_and_categories(df)
    labels = tf.ragged.constant(y)
    label_binarized = model.encoder(labels).numpy()

    predict = model.predict(x)
    print(x[INDEX])
    print(label_binarized[INDEX])
    print(predict[INDEX])
    print(model.invert_multi_hot(label_binarized[INDEX], ALPHA))
    print(model.invert_multi_hot(predict[INDEX], ALPHA))


if __name__ == "__main__":
    main()
