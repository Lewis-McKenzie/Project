import xml.etree.ElementTree as et
import pandas as pd
import numpy as np
from typing import Dict, Union, List

DIR = "F:\\Documents\\Uni\\PRBX\\Project\\data\\ABSA16_Restaurants_Train_SB1_v2.xml"
EMB_PATH = "F:\\Documents\\Uni\\PRBX\\Project\\data\\word_embeddings\\glove.6B\\glove.6B.100d.txt"

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
                    opinions.append({"target": opinion.attrib["target"],
                                    "category": opinion.attrib["category"],
                                    "polarity": opinion.attrib["polarity"],
                                    "from": opinion.attrib["from"],
                                    "to": opinion.attrib["to"]})
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
