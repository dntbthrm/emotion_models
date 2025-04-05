import pandas as pd
import yaml
import numpy as np
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences

raw_data = pd.read_csv("../../dataset/ru-go-emotions-raw.csv")

emotion_columns = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust",
    "embarrassment", "excitement", "fear", "gratitude", "grief", "joy",
    "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral"
]

raw_data = raw_data[["ru_text"] + emotion_columns].dropna()

def extract_labels(row):
    return [col for col in emotion_columns if row[col] == 1]

raw_data["label"] = raw_data[emotion_columns].apply(extract_labels, axis=1)

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(raw_data["label"])

with open("model_data/label_to_index.pkl", "wb") as f:
    pickle.dump(mlb.classes_, f)

MAX_VOCAB_SIZE = 20000
MAX_SEQUENCE_LENGTH = 100

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(raw_data["ru_text"])

X = tokenizer.texts_to_sequences(raw_data["ru_text"])
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

np.save("model_data/X.npy", X)
np.save("model_data/y.npy", y)

with open("model_data/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Данные сохранены!")
