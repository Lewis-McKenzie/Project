import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils import Loader, Preprocessor
from models import BasicModel
from argumentation import Argument
import tensorflow as tf

DIR = "F:\\Documents\\Uni\\PRBX\\Project\\data\\ABSA16_Restaurants_Train_SB1_v2.xml"
EMB_PATH = "F:\\Documents\\Uni\\PRBX\\Project\\data\\word_embeddings\\glove.6B\\glove.6B.100d.txt"
MODEL_WEIGHTS = "F:\\Documents\\Uni\\PRBX\\Project\\data\\model_weights\\basic_model"

def main() -> None:    
    df = Loader.load(DIR)
    processor = Preprocessor(df)
    train_dataset, validation_dataset = processor.make_polarity_category_dataset(16)

    embeddings_index = Loader.load_word_embedings(EMB_PATH)
    embedding_matrix = processor.make_embedding_matrix(embeddings_index)

    LR = 0.005
    EPOCHS = 5

    model = BasicModel(processor.polarity_category_encoder, embedding_matrix)
    model.adapt_encoder(df["text"])
    model.compile(loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.F1Score()])

    model.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS, verbose=1)
    model.save(MODEL_WEIGHTS)

    ALPHA = 0.5
    INDEX = 0

    predict = model.predict(tf.convert_to_tensor(df["text"]))
    print(df["text"][INDEX])
    print(processor.get_encoded_labels()[INDEX])
    print(predict[INDEX])
    print(model.invert_multi_hot(processor.get_encoded_labels()[INDEX], ALPHA))
    print(model.invert_multi_hot(predict[INDEX], ALPHA))

    CUT = 10

    argument = Argument(model, df["text"].to_list()[:CUT], predict[:CUT], ALPHA)
    #argument = Argument(model, df["text"].to_list()[:CUT], processor.get_encoded_labels()[:CUT], ALPHA)
    fl = argument.fuzzy_labeling(12)

if __name__ == "__main__":
    main()
