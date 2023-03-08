import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from typing import Set, List

from utils import Loader, Preprocessor, DIR
from models import BasicModel

#Spacy
import spacy
nlp = spacy.load('en_core_web_sm')

df = Loader.load(DIR)
#Preprocessor.process_all(df)

model = BasicModel()

txt = df["text"][0]
print(df.shape)
print(df["text"])
model.adapt_encoder(df["text"])
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

'''category = df["opinions"][0][0]["category"]
categories: Set[str] = set()
for opinions in df["opinions"]:
    for opinion in opinions:
        categories.add(opinion["category"])
print(categories)

label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(list(categories))
#integer_category = label_encoder.fit_transform(categories)
#encoded_y = tf.keras.utils.to_categorical(integer_category)

def merge_encodings(encodings: List[List[float]]) -> List[float]:
    if encodings == []:
        return [0.0 for _, _ in enumerate(categories)]
    return [max([encoding[i] for encoding in encodings]) for i, _ in enumerate(categories)]

merged_encodings = []
for opinions in df["opinions"]:
    encoded_opinions = []
    for opinion in opinions:
        encoded_opinions.append(tf.keras.utils.to_categorical(label_encoder.transform([opinion["category"]]), num_classes=len(categories))[0])
    merged_encodings.append(merge_encodings(encoded_opinions))

test_reviews = [review.lower() for review in df["text"]]
test_aspect_terms = []
for review in nlp.pipe(test_reviews):
    chunks = [(chunk.root.text) for chunk in review.noun_chunks if chunk.root.pos_ == 'NOUN']
    test_aspect_terms.append(chunks)

print(test_aspect_terms[0])

print(merged_encodings[3], len(merged_encodings))
#print(encoded_y[0])
print(df.shape)
print(category)
print(txt)
print(model.predict(np.array([txt])))'''

label_encoder = LabelEncoder()
integer_category = label_encoder.fit_transform(df["category"])
encoded_y = tf.keras.utils.to_categorical(integer_category)

model.fit(df["text"], encoded_y, epochs=10, verbose=1)
raw_predictions = model.predict(df["text"])
predicitons = label_encoder.inverse_transform(np.argmax(raw_predictions, axis=-1))
print(predicitons[1], df["category"][1])

#print(label_encoder.inverse_transform(np.argmax(model.predict(["I went to the restaurant and the waiter was horrible"]), axis=-1)))
