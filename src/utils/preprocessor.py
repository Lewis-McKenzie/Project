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
    categories: Set[str]
    word2vec: Word2Vec

    @staticmethod
    def process_all(data: pd.DataFrame) -> Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        Preprocessor.initialize_categories(data["opinions"])
        Preprocessor.initialize_category_encoder()
        Preprocessor.initialize_word2vec(data["text"])
        text: List[List[List[float]]] = []
        aspects: List[List[float]] = []
        categories: List[List[float]] = []
        for _, entry in data.iterrows():
            x, (aspect, category) = Preprocessor.process(entry["text"], entry["opinions"])
            text.append(Preprocessor.pad(x, MAX_LEN, VECTOR_SIZE))
            aspects.append(aspect)
            categories.append(category)
        return tf.convert_to_tensor(text), (tf.convert_to_tensor(aspects), tf.convert_to_tensor(categories))

    @staticmethod
    def initialize_categories(opinions: List[List[Dict[str, str]]]) -> None:
        Preprocessor.categories = set()
        for opinion in opinions:
            for entry in opinion:
                Preprocessor.categories.add(entry["category"])

    @staticmethod
    def initialize_category_encoder() -> None:
        terms = tf.ragged.constant(list(Preprocessor.categories))
        Preprocessor.category_encoder = tf.keras.layers.StringLookup(output_mode="multi_hot")
        Preprocessor.category_encoder.adapt(terms)

    @staticmethod
    def initialize_word2vec(sentences: List[str]) -> None:
        tokenized_sentences = [Preprocessor.tokenize(sentence) for sentence in sentences]
        Preprocessor.word2vec = Word2Vec(tokenized_sentences, min_count=1, vector_size=VECTOR_SIZE)
        Preprocessor.word2vec.train(tokenized_sentences, total_examples=len(tokenized_sentences), epochs=1)

    @staticmethod
    def process(x: str, y: List[Dict[str, str]]) -> Tuple[List[List[float]], Tuple[List[float], List[float]]]:
        words = Preprocessor.remove_stopwords(Preprocessor.tokenize(x))
        return Preprocessor.embed_words(words), Preprocessor.embed_opinions(words, y)

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
    def embed_opinions(words: List[str], opinions: List[Dict[str, str]]) -> Tuple[List[float], List[float]]:
        aspect_embedding = [0.0 for _ in range(MAX_LEN)]
        category_embedding = Preprocessor.category_encoder([opinion["category"] for opinion in opinions])
        for opinion in opinions:
            Preprocessor.embed_aspect(opinion, words, aspect_embedding)
        return aspect_embedding, category_embedding

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
