import tensorflow as tf
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Embedding, LSTM, Dense, Dropout
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.metrics import AUC, Precision, Recall
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pickle
import logging
from keras._tf_keras.keras.callbacks import EarlyStopping

from TextPreprocessor import BASIC_EMOTIONS
tf.config.set_visible_devices([], 'GPU')
# Загружаем данные
X_train = np.load('v10_X_train.npz')['arr_0']
X_val = np.load('v10_X_val.npz')['arr_0']
y_train = np.load('v10_y_train.npz')['arr_0']
y_val = np.load('v10_y_val.npz')['arr_0']

# Рассчитываем веса для классов
# Рассчитываем веса для классов
y_integers = np.argmax(y_train, axis=1)
class_weights = compute_class_weight('balanced', classes=np.unique(y_integers), y=y_integers)
class_weights_dict = dict(enumerate(class_weights))
print(class_weights_dict)
# Загрузка токенизатора и меток
with open('v10_tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
'''
# Считаем веса классов
#class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=np.argmax(y_train, axis=1))
#class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
'''
# Построение модели
'''model = Sequential([
    Embedding(input_dim=20000, output_dim=256, input_length=100),
    LSTM(128, return_sequences=True, recurrent_dropout=0.3),
    LSTM(32),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(len(BASIC_EMOTIONS), activation='sigmoid')
])'''

model = Sequential([
    Embedding(input_dim=20000, output_dim=256, input_length=100),
    LSTM(128, return_sequences=True),
    LSTM(64),
    Dropout(0.2),
    #Dense(32, activation='relu'),
    Dense(len(BASIC_EMOTIONS), activation='sigmoid')
])

#model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy', AUC(), Precision(), Recall()])
# Обучаем модель
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32,
    class_weight=class_weights_dict,
    callbacks=[early_stop]
)
model.save('emotion_model_v12.keras')

# Оценка модели
y_pred = model.predict(X_val)
y_pred = (y_pred > 0.5).astype(int)

# Вычисление F1 score
f1 = f1_score(y_val, y_pred, average=None)
print(f'F1 score по каждому классу: {f1}')

# Средний F1 score
print(f'Средний F1 score: {np.mean(f1)}')
