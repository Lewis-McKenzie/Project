import tensorflow as tf
from typing import List, Dict

from utils import Loader, Preprocessor
from train_sandbox import DIR, EMB_PATH, MODEL_WEIGHTS, TEST_DIR
from models import BasicModel
from argumentation import Argument


def main() -> None:
    ALPHA = 0.5

    acd_model = Loader.load_model(MODEL_WEIGHTS, "acd_model")
    polarity_model = Loader.load_model(MODEL_WEIGHTS, "polarity_model")
    
    test_df = Loader.load(TEST_DIR)
    encoded_predicted_categories = acd_model.predict(test_df["text"])
    predicted_categories = acd_model.invert_all(encoded_predicted_categories, ALPHA)
    text_with_categories = [f"{category} {test_df.text[i]}" for i, categories in enumerate(predicted_categories) for category in categories]
    encoded_predicted_polarities = polarity_model.predict(tf.convert_to_tensor(text_with_categories))

    results: Dict[str, Dict[str, List[float]]] = dict()
    for i, twc in enumerate(text_with_categories):
        text = " ".join(twc.split(" ")[1:])
        category = twc.split(" ")[0]
        if text not in results.keys():
            results[text] = {category: encoded_predicted_polarities[i]}
        else:
            results[text][category] = encoded_predicted_polarities[i]

    argument = Argument(model, test_df["text"].to_list()[:CUT], predict[:CUT], ALPHA)
    #argument = Argument(model, df["text"].to_list()[:CUT], encoded_labels[:CUT], ALPHA)
    fl = argument.fuzzy_labeling(12)

if __name__ == "__main__":
    main()