import numpy as np
import os
from sklearn.model_selection import train_test_split
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils import Loader, Preprocessor, DIR
from models import BasicModel
from argumentation import Argument
import tensorflow as tf

df = Loader.load(DIR)
processor = Preprocessor(df)

#model = BasicModel()
model = BasicModel(processor.polarity_category_encoder)
model.adapt_encoder(df["text"])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

train_dataset, validation_dataset = processor.make_polarity_category_dataset(64)

model.fit(train_dataset, validation_data=validation_dataset, epochs=20, verbose=1)

INDEX = 3
ALPHA = 0.5

text_batch, label_batch = next(iter(train_dataset))
x = text_batch[0]
y = label_batch[0]

print(x)
print(y)
out = model.predict(text_batch)[0]

print(model.invert_multi_hot(out, ALPHA))
print(model.invert_multi_hot(y, ALPHA))

predict = model.predict(tf.convert_to_tensor(df["text"]))

argument = Argument(model, df["text"].to_list(), predict, ALPHA)
attackers = argument.attack(x)
for i, c in enumerate(model.invert_all(predict, ALPHA)):
    #print(c, model.invert_multi_hot(y[i], ALPHA))
    pass
