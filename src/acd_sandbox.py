from utils import Loader, Preprocessor, RESTAURANT_TRAIN_PATH, LAPTOP_TRAIN_PATH, GLOVE_EMBEDINGS_PATH, MODEL_PATH, LAPTOP_TEST_PATH, RESTAURANT_TEST_PATH
from models import BasicModel
import tensorflow as tf

def train(filepath, model_name, test_filepath) -> None:    
    df = Loader.load(filepath)
    processor = Preprocessor(df)

    # ensure test exclusive categories are encoded
    test_df = Loader.load(test_filepath)
    processor.update_categories(list(df["opinions"]) + list(test_df["opinions"]))
    train_dataset, validation_dataset = processor.make_category_dataset(16)

    embeddings_index = Loader.load_word_embedings(GLOVE_EMBEDINGS_PATH)

    LR = 0.005
    EPOCHS = 30

    model = BasicModel(processor.category_encoder, len(processor.get_vocab()), name=model_name)
    model.adapt_encoder(df["text"])
    model.init_embeddings(embeddings_index)
    ALPHA = 0.5
    model.compile(loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                metrics=['accuracy', tf.keras.metrics.Precision(thresholds=ALPHA), tf.keras.metrics.Recall(thresholds=ALPHA), tf.keras.metrics.F1Score(average="micro", threshold=ALPHA)])

    model.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS, verbose=1)
    model.save_model(MODEL_PATH)

    INDEX = 0

    predict = model.predict(df["text"])
    print(df["text"][INDEX])
    print(processor.get_category_encoded_labels()[INDEX])
    print(predict[INDEX])
    print(model.invert_multi_hot(processor.get_category_encoded_labels()[INDEX], ALPHA))
    print(model.invert_multi_hot(predict[INDEX], ALPHA))

def main() -> None:
    train(RESTAURANT_TRAIN_PATH, "acd_rest_model", RESTAURANT_TEST_PATH)
    train(LAPTOP_TRAIN_PATH, "acd_lapt_model", LAPTOP_TEST_PATH)

if __name__ == "__main__":
    main()
