import xml.etree.ElementTree as et
import pandas as pd
import numpy as np
from typing import Dict, Union, List
import tensorflow as tf

from models import BasicModel

class Loader:
    @staticmethod
    def load(filepath: str) -> pd.DataFrame:
        xml_data = et.parse(filepath)
        parsed_xml = Loader.parse_xml(xml_data)
        return pd.DataFrame(parsed_xml)

    @staticmethod
    def parse_xml(xml: et.ElementTree):
        d = []
        for review in xml.iter("Review"):
            for sentence in review.iter("sentence"):
                entry: Dict[str, Union[str, List[Dict[str, str]]]] = {}
                entry["review_id"] = sentence.attrib["id"].split(":")[0]
                text_elem = sentence.find("text")
                if text_elem is None or text_elem.text is None:
                    id = sentence.attrib["id"]
                    raise Exception(f"Text not found for review {id}")
                entry["text"] = text_elem.text.lower()
                opinions = []
                for opinion in sentence.iter("Opinion"):
                    opinions.append({"category": opinion.attrib["category"],
                                     "polarity": opinion.attrib["polarity"]})
                entry["opinions"] = opinions
                d.append(entry)
        return d

    @staticmethod
    def load_word_embedings(filepath: str):
        embeddings_index = {}
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f.readlines():
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                embeddings_index[word] = coefs
        return embeddings_index

    @staticmethod
    def load_model(path: str, name: str) -> BasicModel:
        path += f"\\{name}"
        with open(f"{path}\\categories.txt", 'r') as file:
            cats = file.read()
        encoder = tf.keras.layers.StringLookup(output_mode="multi_hot", num_oov_indices=0)
        encoder.adapt(cats.split(","))
        model = BasicModel(encoder)
        model.pipeline = tf.keras.models.load_model(path)
        return model
