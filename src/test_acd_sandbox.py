import tensorflow as tf

from utils import Loader, Preprocessor
from train_sandbox import MODEL_WEIGHTS, TEST_DIR


def main() -> None:
    model = Loader.load_model(MODEL_WEIGHTS, "acd_model")
    
    test_df = Loader.load(TEST_DIR)
    labels = tf.ragged.constant(Preprocessor.category_values(test_df))
    encoded_labels = model.encoder(labels).numpy()
    ALPHA = 0.5
    LR=0.001
    model.compile(loss='binary_crossentropy',
                    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                    metrics=['binary_accuracy', tf.keras.metrics.Precision(thresholds=ALPHA), tf.keras.metrics.Recall(thresholds=ALPHA), tf.keras.metrics.F1Score(threshold=ALPHA)])
    model.evaluate(test_df["text"], encoded_labels)

    INDEX = 2
    predict = model.predict(test_df["text"])
    print(test_df["text"][INDEX])
    print(encoded_labels[INDEX])
    print(predict[INDEX])
    print(model.invert_multi_hot(encoded_labels[INDEX], ALPHA))
    print(model.invert_multi_hot(predict[INDEX], ALPHA))

if __name__ == "__main__":
    main()