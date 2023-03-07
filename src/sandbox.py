import numpy as np

from utils import Loader, Preprocessor, DIR
from models import BasicModel

df = Loader.load(DIR)
Preprocessor.process_all(df)

model = BasicModel()

txt = df["text"][0]
print(txt)
print(model.predict(np.array([txt])))
