import pandas as pd
import os
import numpy as np
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

raw_data = pd.read_csv("../../dataset/ru-go-emotions-raw.csv")

emotion_columns = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust",
    "embarrassment", "excitement", "fear", "gratitude", "grief", "joy",
    "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral"
]

raw_data = raw_data[["ru_text"] + emotion_columns].dropna()

nltk.download("stopwords")
stop_words = set(stopwords.words("russian"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\S+|#\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text.strip()

raw_data["ru_text"] = raw_data["ru_text"].astype(str).apply(clean_text)

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
if not os.path.exists("model_data/X_train.npy"):
    np.save("model_data/X_train.npy", X_train)
    np.save("model_data/X_test.npy", X_test)
    np.save("model_data/y_train.npy", y_train)
    np.save("model_data/y_test.npy", y_test)
    print("Данные сохранены.")
else:
    print("Данные уже существуют и не перезаписываются.")


print("Данные сохранены!")
