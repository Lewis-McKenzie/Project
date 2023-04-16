import tensorflow as tf

from utils import Loader, Preprocessor
from train_sandbox import DIR, EMB_PATH, MODEL_WEIGHTS
from argumentation import Argument


def main() -> None:
    df = Loader.load(DIR)
    processor = Preprocessor(df)
    model = Loader.load_model(MODEL_WEIGHTS, processor.polarity_category_encoder)
    
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