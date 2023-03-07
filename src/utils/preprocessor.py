from typing import List, Dict
import pandas as pd

class Preprocessor:
    @staticmethod
    def tokenize(text: str) -> List[str]:
        pass

    @staticmethod
    def embed(words: List[str]) -> List[int]:
        pass

    @staticmethod
    def pad(embedded_words: List[int]) -> List[int]:
        pass

    @staticmethod
    def one_hot(embedded_words: List[int]) -> List[List[int]]:
        pass

    @staticmethod
    def process(x: str, y: List[Dict[str, str]]) -> None:
        pass

    @staticmethod
    def process_all(data: pd.DataFrame) -> None:
        for _, entry in data.iterrows():
            Preprocessor.process(entry["text"], entry["opinions"])
