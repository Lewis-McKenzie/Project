import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils import Loader, Preprocessor, DIR
from models import BasicModel

df = Loader.load(DIR)
x, y = Preprocessor.process_all(df)

model = BasicModel()

model.adapt_encoder(df["text"])
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

model.fit(df["text"], y, epochs=10, verbose=1)

INDEX = 0

txt: str = df["text"][INDEX]
aspects = [opinion["target"] for opinion in df["opinions"][INDEX]]
categories = [opinion["category"] for opinion in df["opinions"][INDEX]]
print(txt)
print(aspects)
print(categories)
print(model.predict(np.array([txt.lower()])))
print(y[INDEX])

