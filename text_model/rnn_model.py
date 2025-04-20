import re
import numpy as np
import pandas as pd
import tensorflow as tf
import gensim.downloader as api
from razdel import tokenize
from pymorphy2 import MorphAnalyzer
from nlpaug import Augmenter
from nlpaug.util import Action
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras._tf_keras.keras import layers, models, utils
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import keras._tf_keras.keras.metrics as metrics
from keras._tf_keras.keras import backend as K
#from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
from functools import lru_cache
import pickle
from tqdm import tqdm

from sklearn.metrics import classification_report, confusion_matrix

import logging
import time
from datetime import datetime

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Конфигурация
CONFIG = {
    "max_words": 50000,
    "max_len": 100, #200
    "embed_dim": 300,
    "lstm_units": 256,
    "dense_units": 192,
    "dropout": 0.4,
    "batch_size": 128, #512
    "epochs": 50,
    "lr": 3e-4,
    "focal_gamma": 2.0,
    "aug_multiplier": 2,
    "thresholds": {
        'anger': 0.42,
        'disgust': 0.38,
        'fear': 0.35,
        'joy': 0.45,
        'sadness': 0.41,
        'surprise': 0.33,
        'neutral': 0.47
    }
}


class MultilabelF1:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.precision = tf.keras.metrics.Precision(thresholds=threshold)
        self.recall = tf.keras.metrics.Recall(thresholds=threshold)

    def __call__(self, y_true, y_pred):
        p = self.precision(y_true, y_pred)
        r = self.recall(y_true, y_pred)
        return 2 * ((p * r) / (p + r + 1e-7))

    def get_config(self):
        return {
            "num_classes": self.num_classes,
            "threshold": self.threshold,
            "average": self.average
        }


class DataAugmenter:
    def __init__(self):
        from nlpaug.augmenter.word import ContextualWordEmbsAug

        # Используем встроенные модели вместо загрузки через gensim
        self.aug = ContextualWordEmbsAug(
            model_path='bert-base-uncased',  # Используем BERT вместо Word2Vec
            action="substitute",
            aug_p=0.2,
            device='cpu'
        )

    def augment(self, text, n=2):
        try:
            return [self.aug.augment(text)[0] for _ in range(n)]
        except:
            return [text] * n  # Возвращаем оригинал при ошибке


class EmotionClassifier:
    def __init__(self, config):
        self.config = config
        #self.preprocessor = TextPreprocessor()
        self.augmenter = DataAugmenter()
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=config["max_words"], oov_token="<OOV>")
        self.model = None
        logger.info("Инициализирован классификатор эмоций с конфигурацией: %s", config)

    def _prepare_embeddings(self):
        logger.info("Загрузка FastText эмбеддингов...")
        start_time = time.time()
        try:
            ft = api.load('fasttext-wiki-news-subwords-300')
            logger.info("Эмбеддинги загружены за %.2f сек", time.time() - start_time)
        except Exception as e:
            logger.error("Ошибка загрузки эмбеддингов: %s", str(e))
            raise
        embedding_matrix = np.zeros((self.config["max_words"], 300))
        for word, i in self.tokenizer.word_index.items():
            if i < self.config["max_words"] and word in ft:
                embedding_matrix[i] = ft[word]
        return embedding_matrix

    def _focal_loss(self):
        gamma = self.config["focal_gamma"]

        def loss(y_true, y_pred):
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
            ce = -y_true * tf.math.log(y_pred)
            fl = ce * tf.pow(1 - y_pred, gamma)
            return tf.reduce_mean(fl)

        return loss

    '''def build_model(self):
        inputs = layers.Input(shape=(self.config["max_len"],))

        # Embedding с предобученными весами
        embedding_matrix = self._prepare_embeddings()
        x = layers.Embedding(
            input_dim=self.config["max_words"],
            output_dim=300,
            weights=[embedding_matrix],
            trainable=False,
            mask_zero=True
        )(inputs)

        # Attention LSTM
        x = layers.Bidirectional(layers.LSTM(
            self.config["lstm_units"],
            return_sequences=True,
            recurrent_dropout=0.2
        ))(x)
        x = layers.Attention(use_scale=True)([x, x])
        x = layers.Bidirectional(layers.LSTM(
            self.config["lstm_units"] // 2,
            recurrent_dropout=0.2
        ))(x)

        # Классификатор
        x = layers.Dense(self.config["dense_units"], activation='relu')(x)
        x = layers.Dropout(self.config["dropout"])(x)
        outputs = layers.Dense(7, activation='sigmoid')(x)

        self.model = models.Model(inputs=inputs, outputs=outputs)

        self.model.compile(
            optimizer=Adam(
                learning_rate=self.config["lr"],
                clipnorm=1.0,
                #global_clipnorm=1.0
            ),
            loss=self._focal_loss(),
            metrics=[
                MultilabelF1(threshold=0.5),
                tf.keras.metrics.AUC(name='auc')
            ]
        )'''

    def build_model(self):
        self.model = tf.keras.Sequential([
            layers.Embedding(input_dim=self.config["max_words"], output_dim=128, mask_zero=True),
            layers.LSTM(64, implementation=1),  # implementation=1 отключает CuDNN
            layers.Dense(7, activation='sigmoid')
        ])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        #return model

    def _balance_data(self, X, y):
        ros = RandomOverSampler(random_state=42)
        X_res, y_res = ros.fit_resample(X, y)
        return X_res, y_res

    def train(self, file_path):
        # Загрузка и предобработка
        logger.info("Начало обучения. Загрузка данных из %s", file_path)
        try:
            df = pd.read_csv(file_path)
            # Удаление строк с пропущенными текстами
            df = df[df['ru_text'].notna()]
            # Преобразование всех значений в строки
            df['ru_text'] = df['ru_text'].astype(str)
            logger.info("Данные загружены. Размер: %s", df.shape)
        except Exception as e:
            logger.error("Ошибка загрузки файла: %s", str(e))
            return
        tqdm.pandas()

        # Токенизация
        logger.info("Начало токенизации...")
        start_time = time.time()
        try:
            self.tokenizer.fit_on_texts(df['ru_text'])
            sequences = self.tokenizer.texts_to_sequences(df['ru_text'])
            padded = pad_sequences(
                sequences,
                maxlen=self.config["max_len"],
                padding='post'  # <-- Добавьте этот параметр
            )
            logger.info("Токенизация завершена за %.2f сек", time.time() - start_time)
        except Exception as e:
            logger.error("Ошибка токенизации: %s", str(e))
            return

        # Аугментация
        logger.info("Начало аугментации данных. Множитель: %d", self.config["aug_multiplier"])
        start_time = time.time()
        try:
            texts_aug, labels_aug = [], []
            idx = 1
            for (seq, label) in zip(padded, df.iloc[:, 1:8].values):
                texts_aug.extend([seq] * self.config["aug_multiplier"])
                labels_aug.extend([label] * self.config["aug_multiplier"])
                if idx % 1000 == 0:
                    logger.debug("Обработано %d примеров", idx)
                idx += 1

            X = np.vstack([padded] + texts_aug)
            y = np.vstack([df.iloc[:, 1:8].values] + labels_aug)
            logger.info("Аугментация завершена. Новый размер: %s", X.shape)
        except Exception as e:
            logger.error("Ошибка аугментации: %s", str(e))
            return

        # Балансировка
        '''logger.info("Балансировка данных...")
        start_time = time.time()
        try:
            X_bal, y_bal = self._balance_data(X, y)
            logger.info("Балансировка завершена за %.2f сек. Новый размер: %s",
                        time.time() - start_time, X_bal.shape)
        except Exception as e:
            logger.error("Ошибка балансировки: %s", str(e))
            return'''

        # Разделение данных
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.15, random_state=42
        )

        # Веса классов
        class_weights = compute_class_weight(
            'balanced', classes=np.arange(7), y=y.argmax(axis=1))
        class_weights = dict(enumerate(class_weights))

        # Обучение
        logger.info("Компиляция модели...")
        try:
            self.build_model()
            logger.info(self.model.summary())
        except Exception as e:
            logger.error("Ошибка компиляции модели: %s", str(e))
            return

        logger.info("Начало обучения модели...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config["epochs"],
            batch_size=self.config["batch_size"],
            class_weight=class_weights,
            callbacks=[
                EarlyStopping(patience=7, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)
            ],
            verbose=2
        )
        logger.info("Обучение завершено. История обучения: %s", history.history)
        return history

        '''  def f1_score(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            recall = true_positives / (possible_positives + K.epsilon())
            return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

        logger.info("Оптимизация порогов классификации...")
        try:
            y_pred = self.model.predict(X_val, verbose=0)
            for i, emotion in enumerate(['anger', 'disgust', 'fear', 'joy',
                                         'sadness', 'surprise', 'neutral']):
                best_thresh = 0.0
                best_f1 = 0.0
                logger.debug("Поиск порога для %s...", emotion)
                for thresh in np.linspace(0.3, 0.6, 31):
                    f1 = f1_score(y_val[:, i], (y_pred[:, i] > thresh).astype(int))
                    if f1 > best_f1:
                        best_f1 = f1
                        best_thresh = thresh
                self.config['thresholds'][emotion] = best_thresh
                logger.info("Лучший порог для %s: %.3f (F1: %.4f)", emotion, best_thresh, best_f1)
        except Exception as e:
            logger.error("Ошибка оптимизации порогов: %s", str(e))'''


    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_pred_bin = np.zeros_like(y_pred)
        for i, thresh in enumerate(self.config['thresholds'].values()):
            y_pred_bin[:, i] = (y_pred[:, i] > thresh).astype(int)

        print(classification_report(
            y_test, y_pred_bin,
            target_names=self.config['thresholds'].keys(),
            digits=4
        ))


