import os

PROJECT_PATH = os.getcwd()
DATA_PATH = os.path.join(PROJECT_PATH, "data")
RESTAURANT_TRAIN_PATH = os.path.join(DATA_PATH, "ABSA16_Restaurants_Train_SB1_v2.xml")
RESTAURANT_TEST_PATH = os.path.join(DATA_PATH, "EN_REST_SB1_TEST.xml")
LAPTOP_TRAIN_PATH = os.path.join(DATA_PATH, "ABSA16_Laptops_Train_SB1_v2.xml")
LAPTOP_TEST_PATH = os.path.join(DATA_PATH, "EN_LAPT_SB1_TEST_.xml")
GLOVE_EMBEDINGS_PATH = os.path.join(DATA_PATH, "word_embeddings\\glove.6B\\glove.6B.100d.txt")
MODEL_PATH = os.path.join(DATA_PATH, "model_weights")
