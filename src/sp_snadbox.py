from utils import Loader, Preprocessor, RESTAURANT_TRAIN_PATH, LAPTOP_TRAIN_PATH, GLOVE_EMBEDINGS_PATH, MODEL_PATH
from models import BasicModel
import tensorflow as tf

def train(filepath, model_name) -> None:    
    df = Loader.load(filepath)
    processor = Preprocessor(df)
    train_dataset, validation_dataset = processor.make_polarity_dataset(16)

    embeddings_index = Loader.load_word_embedings(GLOVE_EMBEDINGS_PATH)

    LR = 0.005
    EPOCHS = 5

    model = BasicModel(processor.polarity_encoder, len(processor.get_vocab()), name=model_name)
    model.adapt_encoder(list(df["text"]) + processor.categories)
    model.init_embeddings(embeddings_index)
    ALPHA = 0.5
    model.compile(loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                metrics=['accuracy', tf.keras.metrics.Precision(thresholds=ALPHA), tf.keras.metrics.Recall(thresholds=ALPHA), tf.keras.metrics.F1Score(threshold=ALPHA)])
    model.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS, verbose=1)
    model.save_model(MODEL_PATH)

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

def main() -> None:
    train(RESTAURANT_TRAIN_PATH, "sp_rest_model")
    train(LAPTOP_TRAIN_PATH, "sp_lapt_model")

if __name__ == "__main__":
    main()
