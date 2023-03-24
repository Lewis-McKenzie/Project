import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils import Loader, Preprocessor, DIR
from models import BasicModel, PolarityCategoryModel

df = Loader.load(DIR)
x, [y_aspect, y_category, y_polarity_category] = Preprocessor.process_all(df)

#y = y_category
y = y_polarity_category

#model = BasicModel()
model = PolarityCategoryModel()
model.adapt_encoder(df["text"])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

model.fit(df["text"], y, epochs=10, verbose=1)

INDEX = 3

txt: str = df["text"][INDEX]
aspects = [opinion["target"] for opinion in df["opinions"][INDEX]]
categories = [opinion["category"] for opinion in df["opinions"][INDEX]]
print(txt)
print(aspects)
print(categories)
out = model.predict(np.array([df["text"][INDEX]]))
print(out)
print(y[INDEX])

print(model.invert_multi_hot(*out))
print(model.invert_multi_hot(y[INDEX]))
