from typing import List, Dict, Set, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import nltk

MAX_LEN = 96
VECTOR_SIZE = 64

class Preprocessor:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.categories = self.init_categories(data["opinions"])
        self.polarity_categories = self.init_polarity_categories(self.categories)
        self.category_encoder = self.init_encoder(self.categories)
        self.polarity_encoder = self.init_encoder(["positive", "negative", "neutral"])
        self.polarity_category_encoder = self.init_encoder(self.polarity_categories)

    def get_vocab(self) -> List[str]:
        vocab = set()
        for x in self.data["text"]:
            for t in nltk.tokenize.word_tokenize(x):
                vocab.add(t)
        return list(vocab)

    def init_categories(self, opinions: List[List[Dict[str, str]]]) -> List[str]:
        categories = set()
        for opinion in opinions:
            for entry in opinion:
                categories.add(entry["category"])
        return list(categories)

    def init_polarity_categories(self, categories: List[str]) -> List[str]:
        return [f"{polarity} {category}" for polarity in ["positive", "negative", "neutral"] for category in categories]

    def init_encoder(self, options: List[str]) -> tf.keras.layers.StringLookup:
        values = tf.ragged.constant(options)
        encoder = tf.keras.layers.StringLookup(output_mode="multi_hot")
        encoder.adapt(values)
        return encoder

    def make_category_dataset(self, batch_size: int, test_size=0.1) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        x = self.data["text"]
        y = [[opinion["category"] for opinion in opinions] for opinions in self.data["opinions"]]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)#, stratify=y)
        train_dataset = self.make_dataset(x_train, y_train, self.category_encoder, batch_size)
        test_dataset = self.make_dataset(x_test, y_test, self.category_encoder, batch_size, is_train=False)
        return train_dataset, test_dataset

    def make_polarity_dataset(self, batch_size: int, test_size=0.1) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        x, y = self.pair_text_and_categories()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)#, stratify=y)
        train_dataset = self.make_dataset(x_train, y_train, self.polarity_encoder, batch_size)
        test_dataset = self.make_dataset(x_test, y_test, self.polarity_encoder, batch_size, is_train=False)
        return train_dataset, test_dataset

    def pair_text_and_categories(self):
        x = []
        y = []
        for i, opinions in enumerate(self.data["opinions"]):
            cat_pol: Dict[str, Set[str]] = dict()
            for opinion in opinions:
                if not opinion["category"] in opinion:
                    cat_pol[opinion["category"]] = {opinion["polarity"]}
                else:
                    cat_pol[opinion["category"]].add(opinion["polarity"])
            for cat, pols in cat_pol.items():
                y.append(list(pols))
                x.append(cat + " " + self.data["text"][i])
        return x, y

    def make_polarity_category_dataset(self, batch_size: int, test_size=0.1) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        x_train, y_train, x_test, y_test = self.split_polarity_category(test_size)
        train_dataset = self.make_dataset(x_train, y_train, self.polarity_category_encoder, batch_size)
        test_dataset = self.make_dataset(x_test, y_test, self.polarity_category_encoder, batch_size, is_train=False)
        return train_dataset, test_dataset

    def make_dataset(self, x, y, encoder, batch_size: int, is_train=True) -> tf.data.Dataset:
        labels = tf.ragged.constant(y)
        label_binarized = encoder(labels).numpy()
        dataset = tf.data.Dataset.from_tensor_slices(
            (x, label_binarized)
        )
        dataset = dataset.shuffle(batch_size * 10) if is_train else dataset
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def split_polarity_category(self, test_size: float):
        label_binarized = self.get_polarity_category_encoded_labels()
        train_df, test_df = train_test_split(self.data, test_size=test_size, stratify=label_binarized[:, 1])
        x_train, x_test = train_df["text"], test_df["text"]
        y_train, y_test = Preprocessor.polarity_category_values(train_df), Preprocessor.polarity_category_values(test_df)
        return x_train, y_train, x_test, y_test

    def get_polarity_category_encoded_labels(self):
        labels = tf.ragged.constant(Preprocessor.polarity_category_values(self.data))
        return self.polarity_category_encoder(labels).numpy()
    
    def get_category_encoded_labels(self):
        labels = tf.ragged.constant(Preprocessor.category_values(self.data))
        return self.category_encoder(labels).numpy()

    @staticmethod
    def polarity_category_values(df: pd.DataFrame) -> List[List[str]]:
        return [["{} {}".format(opinion["polarity"], opinion["category"]) for opinion in opinions] for opinions in df["opinions"]]
    
    @staticmethod
    def category_values(df: pd.DataFrame) -> List[List[str]]:
        return [[opinion["category"] for opinion in opinions] for opinions in df["opinions"]]
