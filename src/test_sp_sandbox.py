import tensorflow as tf

from utils import Loader, Preprocessor
from train_sandbox import DIR, EMB_PATH, MODEL_WEIGHTS, TEST_DIR
from models import BasicModel
from argumentation import Argument


def main() -> None:
    model = Loader.load_model(MODEL_WEIGHTS, "polarity_model")
    
    test_df = Loader.load(TEST_DIR)
    x, y = Preprocessor.pair_text_and_categories(test_df)
    x = tf.convert_to_tensor(x)
    labels = tf.ragged.constant(y)
    encoded_labels = model.encoder(labels).numpy()
    ALPHA = 0.5
    LR=0.001
    model.compile(loss='binary_crossentropy',
                    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                    metrics=['accuracy', tf.keras.metrics.Precision(thresholds=ALPHA), tf.keras.metrics.Recall(thresholds=ALPHA), tf.keras.metrics.F1Score(threshold=ALPHA)])
    model.evaluate(x, encoded_labels)

    INDEX = 2
    predict = model.predict(x)
    print(x[INDEX])
    print(encoded_labels[INDEX])
    print(predict[INDEX])
    print(model.invert_multi_hot(encoded_labels[INDEX], ALPHA))
    print(model.invert_multi_hot(predict[INDEX], ALPHA))

    CUT = 10

    argument = Argument(model, test_df["text"].to_list()[:CUT], predict[:CUT], ALPHA)
    #argument = Argument(model, df["text"].to_list()[:CUT], encoded_labels[:CUT], ALPHA)
    fl = argument.fuzzy_labeling(12)

if __name__ == "__main__":
    main()