import numpy as np
import pandas as pd
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

txt = df["text"][3]
print(df.shape)
print(df["text"])
model.adapt_encoder(df["text"])
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

x, y = Preprocessor.process_all(df)

test_aspect_terms = []
for review in nlp.pipe(df["text"]):
    chunks = [(chunk.root.text) for chunk in review.noun_chunks if chunk.root.pos_ == 'NOUN']
    test_aspect_terms.append(chunks)

model.fit(df["text"], y, epochs=10, verbose=1)

print(test_aspect_terms[0])
print(y[0], len(y))
print(df.shape)
print(txt)
print(model.predict(np.array([txt])), y[3])

