import numpy as np
import os
from sklearn.model_selection import train_test_split
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils import Loader, Preprocessor, DIR
from models import BasicModel
from argumentation import Argument

df = Loader.load(DIR)
processor = Preprocessor(df)

#model = BasicModel()
model = BasicModel(processor.polarity_category_encoder)
model.adapt_encoder(df["text"])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

train_dataset, validation_dataset = processor.make_polarity_category_dataset(64)

model.fit(train_dataset, validation_data=validation_dataset, epochs=20, verbose=1)

INDEX = 3

txt: str = df["text"][INDEX]
aspects = [opinion["target"] for opinion in df["opinions"][INDEX]]
categories = [opinion["category"] for opinion in df["opinions"][INDEX]]
print(txt)
print(aspects)
print(categories)
out = model.predict(df["text"])
print(out[INDEX])
print(y[INDEX])
ALPHA = 0.5

print(model.invert_multi_hot(out[INDEX], ALPHA))
print(model.invert_multi_hot(y[INDEX], ALPHA))

argument = Argument(model, df["text"].to_list(), out, ALPHA)
attackers = argument.attack(txt)
for i, c in enumerate(model.invert_all(out, ALPHA)):
    #print(c, model.invert_multi_hot(y[i], ALPHA))
    pass
