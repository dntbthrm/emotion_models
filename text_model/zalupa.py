import pandas as pd
import numpy as np
import re
import json
import pickle
import logging

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences

import pymorphy3
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# Класс для предобработки текста
class TextPreprocessor:
    def __init__(self, remove_stopwords=True, remove_punctuation=True):
        self.morph = pymorphy3.MorphAnalyzer()
        self.stopwords = set(stopwords.words("russian")) if remove_stopwords else set()
        self.remove_punctuation = remove_punctuation

    def normalize(self, text):
        text = text.lower()
        if self.remove_punctuation:
            text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\d+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def lemmatize(self, text):
        tokens = text.split()
        lemmatized = [
            self.morph.parse(token)[0].normal_form
            for token in tokens
            if token not in self.stopwords
        ]
        return " ".join(lemmatized)

    def preprocess_text(self, text):
        return self.lemmatize(self.normalize(text))


# Основной пайплайн обработки и разбиения
def preprocess_dataset(csv_path, out_prefix="sran", test_size=0.1, val_size=0.1, max_words=10000, max_len=70):
    logging.info("Загружаем датасет и начинаем препроцессинг...")

    df = pd.read_csv(csv_path)
    assert 'ru_text' in df.columns, "Ожидается колонка 'ru_text' в датасете"

    processor = TextPreprocessor(remove_stopwords=True, remove_punctuation=False)
    tqdm.pandas()
    df['processed_text'] = df['ru_text'].progress_apply(processor.preprocess_text)
    df = df[df['processed_text'].str.strip().astype(bool)]

    # Токенизация
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['processed_text'])
    sequences = tokenizer.texts_to_sequences(df['processed_text'])
    X = pad_sequences(sequences, maxlen=max_len)

    BASIC_EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']
    y = df[BASIC_EMOTIONS].values.astype(np.float32)

    # Разделение
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size / (1 - test_size), random_state=42)

    # Сохраняем токенизатор и конфиг
    with open(f"{out_prefix}_tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    with open(f"{out_prefix}_label_columns.pkl", "wb") as f:
        pickle.dump(BASIC_EMOTIONS, f)

    with open(f"{out_prefix}_config.json", "w", encoding="utf-8") as f:
        json.dump({"max_words": max_words, "max_len": max_len}, f, ensure_ascii=False, indent=4)

    np.save(f"{out_prefix}_X_train.npy", X_train)
    np.save(f"{out_prefix}_X_val.npy", X_val)
    np.save(f"{out_prefix}_X_test.npy", X_test)
    np.save(f"{out_prefix}_y_train.npy", y_train)
    np.save(f"{out_prefix}_y_val.npy", y_val)
    np.save(f"{out_prefix}_y_test.npy", y_test)
    logging.info("Препроцессинг завершён.")

    return X_train, X_val, X_test, y_train, y_val, y_test, tokenizer

X_train, X_val, X_test, y_train, y_val, y_test, tokenizer = preprocess_dataset("../../dataset/ru-go-emotions-v404.csv")
