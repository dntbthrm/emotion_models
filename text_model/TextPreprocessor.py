import re
import os
import json
import pickle
import random
import logging
import numpy as np
import pandas as pd
from razdel import tokenize
from pymorphy3 import MorphAnalyzer
from functools import lru_cache
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
import nlpaug.augmenter.word as naw
import tensorflow as tf
from keras._tf_keras.keras import backend as K
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASIC_EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']

class TextPreprocessor:
    def __init__(self, config=None):
        default_config = {
            'remove_urls': True,
            'replace_emoticons': True,
            'normalize_repeats': True,
            'use_lemmatization': True,
            'stopwords': self._russian_stopwords(),
            'emoticon_map': self._default_emoticon_map(),
            'max_length': 300,
            'min_word_length': 2,
            'augmentation_prob': 0.3,
            'augmenter_type': 'synonym'
        }
        self.config = {**default_config, **(config or {})}
        self.morph = MorphAnalyzer()
        self.token_cache = {}
        self.augmenter = self._init_augmenter(self.config['augmenter_type'])

    def _init_augmenter(self, aug_type):
        if aug_type == 'synonym':
            # Используем аугментатор, который не требует POS-теггинга
            return naw.ContextualWordEmbsAug(model_path='DeepPavlov/rubert-base-cased', action="substitute")
        else:
            logging.warning(f"Неизвестный тип аугментатора: {aug_type}, используется 'synonym'")
            return naw.ContextualWordEmbsAug(model_path='DeepPavlov/rubert-base-cased', action="substitute")


    @staticmethod
    def _russian_stopwords():
        return {'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а',
                'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же',
                'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от',
                'меня', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг',
                'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас',
                'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 'потом', 'себя', 'ничего'}

    @staticmethod
    def _default_emoticon_map():
        return {
            r'[:=]-?\)': ' смайлик_радость ',
            r'[:=]-?\(': ' смайлик_грусть ',
            r'[:=]-?d': ' смайлик_смех ',
            r'[:=]-?\*': ' смайлик_поцелуй ',
            r';-?\)': ' смайлик_подмиг ',
            r'\*_\*': ' смайлик_восхищение '
        }

    @lru_cache(maxsize=100000)
    def _lemmatize(self, word):
        return self.morph.parse(word)[0].normal_form

    def _tokenize(self, text):
        if text in self.token_cache:
            return self.token_cache[text]
        tokens = [token.text.lower() for token in tokenize(text)]
        self.token_cache[text] = tokens
        return tokens

    def _process_emoticons(self, text):
        for pattern, replacement in self.config['emoticon_map'].items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def _normalize_repeats(self, text):
        return re.sub(r'(.)\1{3,}', r'\1\1', text)

    def preprocess_text(self, text):
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'\d+', '', text)

        if self.config['replace_emoticons']:
            text = self._process_emoticons(text)
        if self.config['normalize_repeats']:
            text = self._normalize_repeats(text)

        tokens = self._tokenize(text)
        processed = []
        for token in tokens:
            if len(token) < self.config['min_word_length']:
                continue
            if token in self.config['stopwords']:
                continue
            if self.config['use_lemmatization']:
                token = self._lemmatize(token)
            processed.append(token)

        text = ' '.join(processed)[:self.config['max_length']]

        if text.strip() and random.random() < self.config['augmentation_prob']:
            try:
                aug = self.augmenter.augment(text)
                if isinstance(aug, list):
                    text = aug[0]
                else:
                    text = aug
            except Exception as e:
                logging.warning(f"Ошибка при аугментации: {e}")

        return text.strip()

    def calculate_class_weights(self, y):
        class_counts = y.sum(axis=0)
        total_samples = len(y)
        weights = total_samples / (len(class_counts) * class_counts)
        return dict(zip(y.columns, weights))


class WeightedBinaryCrossentropy(tf.keras.losses.Loss):
    def __init__(self, class_weights):
        super().__init__()
        self.class_weights = tf.constant(class_weights, dtype=tf.float32)

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
        loss = - (self.class_weights * y_true * tf.math.log(y_pred) +
                  (1 - y_true) * tf.math.log(1 - y_pred))
        return tf.reduce_mean(loss)

def make_json_serializable(config):
    """
    Преобразует значения set → list для сериализации в JSON.
    """
    serializable = {}
    for k, v in config.items():
        if isinstance(v, set):
            serializable[k] = list(v)
        else:
            serializable[k] = v
    return serializable


def run_preprocessing_pipeline(input_path, out_prefix="processed"):
    logging.info("Старт препроцессинга...")
    processor = TextPreprocessor()
    df = pd.read_csv(input_path)
    tqdm.pandas()
    df['processed_text'] = df['ru_text'].progress_apply(processor.preprocess_text)
    df = df[df['processed_text'].str.strip().astype(bool)]  # удаляем пустые строки

    tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['processed_text'])

    # Сохраняем токенизатор
    with open(f"{out_prefix}_tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    with open(f"{out_prefix}_label_columns.pkl", "wb") as f:
        pickle.dump(BASIC_EMOTIONS, f)

    with open(f"{out_prefix}_config.json", "w", encoding="utf-8") as f:
        json.dump(make_json_serializable(processor.config), f, ensure_ascii=False, indent=4)

    # Преобразуем текст в последовательности чисел
    X = tokenizer.texts_to_sequences(df['processed_text'])
    X = pad_sequences(X, maxlen=100)
    y = df[BASIC_EMOTIONS].values

    # Балансировка классов
    #ros = RandomOverSampler(random_state=42)
   # X_resampled, y_resampled = ros.fit_resample(X, y)

    np.savez_compressed(f"{out_prefix}_X_resampled.npz", X)
    np.savez_compressed(f"{out_prefix}_y_resampled.npz", y)

    # Разделение на обучающие и валидационные выборки
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

    np.savez_compressed(f"{out_prefix}_X_train.npz", X_train)
    np.savez_compressed(f"{out_prefix}_X_val.npz", X_val)
    np.savez_compressed(f"{out_prefix}_y_train.npz", y_train)
    np.savez_compressed(f"{out_prefix}_y_val.npz", y_val)

    # Сохраняем CSV файлы
    df_train = pd.DataFrame(X_train)
    df_train['label'] = list(y_train)
    df_train.to_csv(f"{out_prefix}_train.csv", index=False)

    df_val = pd.DataFrame(X_val)
    df_val['label'] = list(y_val)
    df_val.to_csv(f"{out_prefix}_val.csv", index=False)

    logging.info(f"✅ Препроцессинг завершен. Сохранены файлы с префиксом '{out_prefix}'")
    return processor, tokenizer, (X_train, y_train), (X_val, y_val)


def load_tokenizer_and_config(out_prefix="processed"):
    with open(f"{out_prefix}_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open(f"{out_prefix}_config.json", "r", encoding='utf-8') as f:
        config = json.load(f)

    return tokenizer, config


def create_dataset(X, y, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(buffer_size=10000).batch(batch_size)
    return dataset


# Пример вызова
if __name__ == "__main__":
    processor, tokenizer, (X_train, y_train), (X_val, y_val) = run_preprocessing_pipeline("../../dataset/ru-go-emotions-preprocessed_v8.csv", "v10")

    # Создание датасетов для обучения
    train_dataset = create_dataset(X_train, y_train)
    val_dataset = create_dataset(X_val, y_val)
