import os
import pandas as pd
import re
import nltk
import numpy as np
import tensorflow
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import pickle

nltk.download("stopwords")
stop_words = set(stopwords.words("russian"))

data = pd.read_csv("../../dataset/ru-go-emotions-raw.csv")
print("Датасет загружен:", data.head())

# Убираем ненужные колонки
text_column = "text"
emotion_columns = data.columns[10:]  # Все эмоции начинаются с 10-го столбца

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\S+|#\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text.strip()

data["clean_text"] = data[text_column].astype(str).apply(clean_text)

# Формируем `y` в one-hot формате
y = data[emotion_columns].values  # Получаем массив 0 и 1

# Создаём токенизатор
MAX_NUM_WORDS = 20000
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<OOV>")

# Загружаем токенизатор, если он уже существует
tokenizer_path = "model_data/tokenizer.pkl"
if os.path.exists(tokenizer_path):
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    print("Токенизатор загружен из файла.")
else:
    tokenizer.fit_on_texts(data["clean_text"])
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)
    print("Токенизатор сохранен в файл.")

sequences = tokenizer.texts_to_sequences(data["clean_text"])
MAX_SEQUENCE_LENGTH = int(np.percentile([len(seq) for seq in sequences], 95))
X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="post")

# Разделяем данные
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Сохраняем данные, если они еще не сохранены
if not os.path.exists("model_data/X_train.npy"):
    np.save("model_data/X_train.npy", X_train)
    np.save("model_data/X_test.npy", X_test)
    np.save("model_data/y_train.npy", y_train)
    np.save("model_data/y_test.npy", y_test)
    print("Данные сохранены.")
else:
    print("Данные уже существуют и не перезаписываются.")

# Сохраняем метки эмоций, если файл не существует
label_to_index_path = "model_data/label_to_index.pkl"
if not os.path.exists(label_to_index_path):
    with open(label_to_index_path, "wb") as f:
        pickle.dump(list(emotion_columns), f)
    print("Метки эмоций сохранены.")
else:
    print("Метки эмоций уже сохранены.")
