import numpy as np
import os
from sklearn.model_selection import train_test_split
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils import Loader, Preprocessor, DIR, EMB_PATH
from models import BasicModel
from argumentation import Argument
import tensorflow as tf

df = Loader.load(DIR)
processor = Preprocessor(df)
train_dataset, validation_dataset = processor.make_polarity_category_dataset(64)

embeddings_index = Loader.load_word_embedings(EMB_PATH)



voc = processor.get_vocab()
num_tokens = len(voc) + 2
embedding_dim = 100
word_index = dict(zip(voc, range(len(voc))))
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector

#model = BasicModel()
model = BasicModel(processor.polarity_category_encoder, embedding_matrix)
model.adapt_encoder(df["text"])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


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
