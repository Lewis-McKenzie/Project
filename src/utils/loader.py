import xml.etree.ElementTree as et
import pandas as pd

DIR = "F:\\Documents\\Uni\\PRBX\\Project\\data\\ABSA16_Restaurants_Train_SB1_v2.xml"

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
                entry = {}
                entry["review_id"] = sentence.attrib["id"].split(":")[0]
                entry["text"] = sentence.find("text").text
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
