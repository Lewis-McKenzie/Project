from typing import List, Dict, Set, Tuple
import pandas as pd
import numpy as np
import tensorflow as tf
import gensim
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

MAX_LEN = 96
VECTOR_SIZE = 64

class Preprocessor:
    category_encoder: tf.keras.layers.StringLookup
    polarity_category_encoder: tf.keras.layers.StringLookup
    categories: Set[str]
    polarity_categories: List[str]
    word2vec: Word2Vec

    @staticmethod
    def process_all(data: pd.DataFrame) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        Preprocessor.initialize_categories(data["opinions"])
        Preprocessor.initialize_encoders()
        Preprocessor.initialize_word2vec(data["text"])
        text: List[List[List[float]]] = []
        aspects: List[List[float]] = []
        categories: List[List[float]] = []        
        polarity_categories: List[List[float]] = []
        for _, entry in data.iterrows():
            x, [aspect, category, polarity_category] = Preprocessor.process(entry["text"], entry["opinions"])
            text.append(Preprocessor.pad(x, MAX_LEN, VECTOR_SIZE))
            aspects.append(aspect)
            categories.append(category)
            polarity_categories.append(polarity_category)
        return tf.convert_to_tensor(text), [tf.convert_to_tensor(target) for target in [aspects, categories, polarity_categories]]

    @staticmethod
    def initialize_categories(opinions: List[List[Dict[str, str]]]) -> None:
        Preprocessor.categories = set()
        for opinion in opinions:
            for entry in opinion:
                Preprocessor.categories.add(entry["category"])
        Preprocessor.polarity_categories = [f"{polarity} {category}" for polarity in ["positive", "negative", "neutral"] for category in Preprocessor.categories]

    @staticmethod
    def initialize_encoders() -> None:
        categories = tf.ragged.constant(list(Preprocessor.categories))
        Preprocessor.category_encoder = tf.keras.layers.StringLookup(output_mode="multi_hot")
        Preprocessor.category_encoder.adapt(categories)
        polarity_categories = tf.ragged.constant(Preprocessor.polarity_categories)
        Preprocessor.polarity_category_encoder = tf.keras.layers.StringLookup(output_mode="multi_hot")
        Preprocessor.polarity_category_encoder.adapt(polarity_categories)

    @staticmethod
    def initialize_word2vec(sentences: List[str]) -> None:
        tokenized_sentences = [Preprocessor.tokenize(sentence) for sentence in sentences]
        Preprocessor.word2vec = Word2Vec(tokenized_sentences, min_count=1, vector_size=VECTOR_SIZE)
        Preprocessor.word2vec.train(tokenized_sentences, total_examples=len(tokenized_sentences), epochs=1)

    @staticmethod
    def process(x: str, y: List[Dict[str, str]]) -> Tuple[List[List[float]], List[List[float]]]:
        words = Preprocessor.remove_stopwords(Preprocessor.tokenize(x))
        return Preprocessor.embed_words(words), [Preprocessor.get_aspect_embedding(words, y), Preprocessor.get_category_embedding(y), Preprocessor.get_polarity_category_embeddding(y)]

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return nltk.word_tokenize(text)

    @staticmethod
    def remove_stopwords(words: List[str]) -> List[str]:
        return [token for token in words if not token in stopwords.words("english")]

    @staticmethod
    def embed_words(words: List[str]) -> List[List[float]]:
        return [Preprocessor.word2vec.wv[word] for word in words]

    @staticmethod
    def get_category_embedding(opinions: List[Dict[str, str]]) -> List[float]:
        return Preprocessor.category_encoder([opinion["category"] for opinion in opinions])

    @staticmethod
    def get_polarity_category_embeddding(opinions: List[Dict[str, str]]) -> List[float]:
        return Preprocessor.polarity_category_encoder(["{} {}".format(opinion["polarity"], opinion["category"]) for opinion in opinions])

    @staticmethod
    def get_aspect_embedding(words: List[str], opinions: List[Dict[str, str]]) -> List[float]:
        aspect_embedding = [0.0 for _ in range(MAX_LEN)]
        for opinion in opinions:
            Preprocessor.embed_aspect(opinion, words, aspect_embedding)
        return aspect_embedding

    @staticmethod
    def embed_aspect(opinion: Dict[str, str], words: List[str], embedding: List[float]) -> None:
        aspect = nltk.word_tokenize(opinion["target"].lower())
        aspect_start_value, aspect_continued_value = Preprocessor.get_polarity_val(opinion["polarity"])            
        count = 0
        for i, word in enumerate(words):
            if word == aspect[0] and count == 0:
                embedding[i] = aspect_start_value
                count += 1
            elif count > 0 and count < len(aspect):
                embedding[i] = aspect_continued_value
                count += 1

    @staticmethod
    def get_polarity_val(polarity: str) -> Tuple[float, float]:
        if polarity == "positive":
            return 2.0, 1.0
        elif polarity == "negative":
            return -2.0, -1.0
        else:
            return 0.5, -0.5

    @staticmethod
    def pad(l: List[List[float]], n: int, m: int) -> List[List[float]]:
        return [[0.0 for _ in range(m)] if i >= len(l) else l[i] for i in range(n)]
