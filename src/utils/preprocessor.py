from typing import List, Dict, Set, Tuple
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

class Preprocessor:
    label_encoder: LabelEncoder
    categories: Set[str]

    @staticmethod
    def process_all(data: pd.DataFrame) -> Tuple[List[str], tf.Tensor]:
        Preprocessor.initialize_categories(data["opinions"])
        Preprocessor.initialize_label_encoder()
        text: List[str] = []
        opinion: List[List[float]] = []
        for _, entry in data.iterrows():
            x, y = Preprocessor.process(entry["text"], entry["opinions"])
            text.append(x)
            opinion.append(y)
        return text, tf.convert_to_tensor(opinion)

    @staticmethod
    def initialize_label_encoder() -> None:
        Preprocessor.label_encoder = LabelEncoder().fit(list(Preprocessor.categories))

    @staticmethod
    def initialize_categories(opinions: List[List[Dict[str, str]]]) -> None:
        Preprocessor.categories = set()
        for opinion in opinions:
            for entry in opinion:
                Preprocessor.categories.add(entry["category"])

    @staticmethod
    def process(x: str, y: List[Dict[str, str]]) -> Tuple[str, List[float]]:
        return x.lower(), Preprocessor.embed_opinions(y)

    @staticmethod
    def embed_opinions(opinions: List[Dict[str, str]]) -> List[float]:
        encoded_opinions = []
        for opinion in opinions:
            encoded_label = Preprocessor.label_encoder.transform([opinion["category"]])
            categorised_label = tf.keras.utils.to_categorical(encoded_label, num_classes=len(Preprocessor.categories))[0]
            encoded_opinions.append(categorised_label)
        return Preprocessor.merge_encodings(encoded_opinions)

    @staticmethod
    def merge_encodings(encodings: List[List[float]]) -> List[float]:
        if encodings == []:
            return [0.0 for _, _ in enumerate(Preprocessor.categories)]
        return [max([encoding[i] for encoding in encodings]) for i, _ in enumerate(Preprocessor.categories)]

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return []

    @staticmethod
    def embed_words(words: List[str]) -> List[float]:
        return []


    @staticmethod
    def pad(embedded_words: List[int]) -> List[int]:
        return []

    @staticmethod
    def one_hot(embedded_words: List[int]) -> List[List[int]]:
        return []
