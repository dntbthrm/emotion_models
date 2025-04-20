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
import model_config as mc
from nltk.stem import WordNetLemmatizer
from string import punctuation
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

#proc_data = pd.read_csv("../../dataset/ru-go-emotions-preprocessed.csv")
proc_data = pd.read_csv("../../dataset/ru-go-emotions-preprocessed_v6.csv")
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = stopwords.words('russian') + list(punctuation)
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    '''text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\S+|#\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text.strip()'''
    words = nltk.word_tokenize(text.lower(), language='russian', preserve_line=True)
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

def extract_labels(row):
    return [col for col in basic_emotions if row[col] == 1]


proc_data["ru_text"] = proc_data["ru_text"].astype(str).apply(clean_text)

proc_data["label"] = proc_data[basic_emotions].apply(extract_labels, axis=1)

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(proc_data["label"])

with open("v6/label_to_index_small.pkl", "wb") as f:
    pickle.dump(mlb.classes_, f)


tokenizer = Tokenizer(num_words=mc.MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(proc_data["ru_text"])

X = tokenizer.texts_to_sequences(proc_data["ru_text"])
X = pad_sequences(X, maxlen=mc.MAX_SEQUENCE_LENGTH)

np.save("v6/X_small.npy", X)
np.save("v6/y_small.npy", y)

with open("v6/tokenizer_small.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in msss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

if not os.path.exists("v6/X_train_small.npy"):
    np.save("v6/X_train_small.npy", X_train)
    np.save("v6/X_test_small.npy", X_test)
    np.save("v6/y_train_small.npy", y_train)
    np.save("v6/y_test_small.npy", y_test)
    print("Данные сохранены.")
else:
    print("Данные уже существуют и не перезаписываются.")

print("Все данные сохранены!")
