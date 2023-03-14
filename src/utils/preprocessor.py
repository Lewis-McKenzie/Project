from typing import List, Dict, Set, Tuple
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

MAX_LEN = 96

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
    def initialize_categories(opinions: List[List[Dict[str, str]]]) -> None:
        Preprocessor.categories = set()
        for opinion in opinions:
            for entry in opinion:
                Preprocessor.categories.add(entry["category"])

    @staticmethod
    def initialize_label_encoder() -> None:
        Preprocessor.label_encoder = LabelEncoder().fit(list(Preprocessor.categories))

    @staticmethod
    def process(x: str, y: List[Dict[str, str]]) -> Tuple[str, List[float]]:
        embedded_aspect = Preprocessor.embed_aspects(x, y)
        embedded_category = Preprocessor.embed_categories(y)
        embedded_opinions = Preprocessor.embed_opinions(x, y)
        return x.lower(), embedded_category

    @staticmethod
    def embed_opinions(text: str, opinions: List[Dict[str, str]]) -> List[List[float]]:
        embedding = [[0.0 for _ in range(MAX_LEN)] for _ in Preprocessor.categories]
        sentence = nltk.word_tokenize(text.lower())
        for opinion in opinions:
            aspect = nltk.word_tokenize(opinion["target"].lower())
            category_index = Preprocessor.label_encoder.transform([opinion["category"]])[0]
            print(Preprocessor.label_encoder.transform([opinion["category"]]))
            polarity = opinion["polarity"]
            if polarity == "positive":
                aspect_start_value = 2.5
                aspect_continued_value = 1.5
            elif polarity == "negative":
                aspect_start_value = -2.5
                aspect_continued_value = -1.5
            else:
                aspect_start_value = 0.5
                aspect_continued_value = -0.5
            count = 0
            for i, word in enumerate(sentence):
                if word == aspect[0] and count == 0:
                    embedding[category_index][i] = aspect_start_value
                    count += 1
                elif count > 0 and count < len(aspect):
                    embedding[category_index][i] = aspect_continued_value
                    count += 1
        return embedding


    @staticmethod
    def embed_categories(opinions: List[Dict[str, str]]) -> List[float]:
        encoded_opinions = []
        for opinion in opinions:
            encoded_label = Preprocessor.label_encoder.transform([opinion["category"]])
            categorised_label = tf.keras.utils.to_categorical(encoded_label, num_classes=len(Preprocessor.categories))[0]
            encoded_opinions.append(categorised_label)
        return Preprocessor.merge_encodings(encoded_opinions)

    @staticmethod
    def merge_encodings(encodings: List[List[float]]) -> List[float]:
        if encodings == []:
            return [0.0 for _ in Preprocessor.categories]
        return [max([encoding[i] for encoding in encodings]) for i, _ in enumerate(Preprocessor.categories)]

    @staticmethod
    def embed_aspects(text: str, opinions: List[Dict[str, str]]) -> List[float]:
        sentence = nltk.word_tokenize(text.lower())
        embedded_aspect = [0.0 for _ in sentence]
        for opinion in opinions:
            aspect = nltk.word_tokenize(opinion["target"].lower())
            count = 0
            for i, word in enumerate(sentence):
                if word == aspect[0]:
                    embedded_aspect[i] = 2.0
                    count += 1
                elif count > 0 and count < len(aspect):
                    embedded_aspect[i] = 1.0
                    count += 1
        return Preprocessor.pad_aspect(embedded_aspect)

    @staticmethod
    def pad_aspect(embedded_aspect: List[float]) -> List[float]:
        return [embedded_aspect[i] if i < len(embedded_aspect) else 0.0 for i in range(MAX_LEN)]

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return []

    @staticmethod
    def embed_words(words: List[str]) -> List[float]:
        return []

    @staticmethod
    def one_hot(embedded_words: List[int]) -> List[List[int]]:
        return []
