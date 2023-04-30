import tensorflow as tf

from utils import Loader, Preprocessor, MODEL_PATH, RESTAURANT_TEST_PATH, LAPTOP_TEST_PATH


def test(filepath, model_name) -> None:
    print(f"testing {model_name} on {filepath}")
    model = Loader.load_model(MODEL_PATH, model_name)
    
    test_df = Loader.load(filepath)
    labels = tf.ragged.constant(Preprocessor.category_values(test_df))
    encoded_labels = model.encoder(labels).numpy()
    ALPHA = 0.5
    LR=0.001
    model.compile(loss='binary_crossentropy',
                    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                    metrics=['accuracy', tf.keras.metrics.Precision(thresholds=ALPHA), tf.keras.metrics.Recall(thresholds=ALPHA), tf.keras.metrics.F1Score(average="micro", threshold=ALPHA)])
    model.evaluate(test_df["text"], encoded_labels)

    INDEX = 2
    predict = model.predict(test_df["text"])
    print(test_df["text"][INDEX])
    print(encoded_labels[INDEX])
    print(predict[INDEX])
    print(model.invert_multi_hot(encoded_labels[INDEX], ALPHA))
    print(model.invert_multi_hot(predict[INDEX], ALPHA))

def main() -> None:
    test(RESTAURANT_TEST_PATH, "acd_rest_model")
    test(LAPTOP_TEST_PATH, "acd_lapt_model")

if __name__ == "__main__":
    main()