import os
import pandas as pd
import re
import nltk
import numpy as np
import tensorflow
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import pickle
from data_preprocessing import basic_emotions

proc_data = pd.read_csv("../../dataset/ru-go-emotions-preprocessed.csv")

nltk.download("stopwords")
stop_words = set(stopwords.words("russian"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\S+|#\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text.strip()

def extract_labels(row):
    return [col for col in basic_emotions if row[col] == 1]


proc_data["ru_text"] = proc_data["ru_text"].astype(str).apply(clean_text)

proc_data["label"] = proc_data[basic_emotions].apply(extract_labels, axis=1)

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(proc_data["label"])

with open("model_data/label_to_index_small.pkl", "wb") as f:
    pickle.dump(mlb.classes_, f)

MAX_VOCAB_SIZE = 20000
MAX_SEQUENCE_LENGTH = 100

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(proc_data["ru_text"])

X = tokenizer.texts_to_sequences(proc_data["ru_text"])
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

np.save("model_data/X_small.npy", X)
np.save("model_data/y_small.npy", y)

with open("model_data/tokenizer_small.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
if not os.path.exists("model_data/X_train_small.npy"):
    np.save("model_data/X_train_small.npy", X_train)
    np.save("model_data/X_test_small.npy", X_test)
    np.save("model_data/y_train_small.npy", y_train)
    np.save("model_data/y_test_small.npy", y_test)
    print("Данные сохранены.")
else:
    print("Данные уже существуют и не перезаписываются.")

print("Данные сохранены!")
