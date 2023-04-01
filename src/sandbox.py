import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils import Loader, Preprocessor, DIR, EMB_PATH
from models import BasicModel
from argumentation import Argument
import tensorflow as tf

df = Loader.load(DIR)
processor = Preprocessor(df)
train_dataset, validation_dataset = processor.make_polarity_category_dataset(64)

embeddings_index = Loader.load_word_embedings(EMB_PATH)
embedding_matrix = processor.make_embedding_matrix(embeddings_index)

model = BasicModel(processor.polarity_category_encoder, embedding_matrix)
model.adapt_encoder(df["text"])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(train_dataset, validation_data=validation_dataset, epochs=30, verbose=1)

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

CUT = 10

argument = Argument(model, df["text"].to_list()[:CUT], predict[:CUT], ALPHA)
#argument = Argument(model, df["text"].to_list()[:CUT], processor.get_encoded_labels()[:CUT], ALPHA)
fl = argument.fuzzy_labeling(120000)


