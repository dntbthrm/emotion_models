from keras._tf_keras.keras.layers import Input, Embedding, LSTM, Dropout, Dense
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.metrics import Precision, Recall, Metric
from keras._tf_keras.keras import backend as K
import tensorflow as tf
import numpy as np

tf.config.set_visible_devices([], 'GPU')
embedding_matrix = np.load("embedding_matrix.npy")
embedding_dim = 300

class F1Score(Metric):
    def __init__(self, name="f1_score", **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = Precision()
        self.recall = Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + K.epsilon()))

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

def weighted_binary_crossentropy(weights):
    weights = tf.constant(weights, dtype=tf.float32)
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        bce = - (weights * y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))
        return K.mean(bce, axis=-1)
    return loss

# Загрузка данных
X_train = np.load("sran_X_train.npy")
X_val = np.load("sran_X_val.npy")
y_train = np.load("sran_y_train.npy")
y_val = np.load("sran_y_val.npy")

# Вычисление весов классов
class_totals = np.sum(y_train, axis=0)
total_labels = np.sum(class_totals)
class_weights = total_labels / (len(class_totals) * class_totals)
loss_fn = weighted_binary_crossentropy(class_weights)

# Построение модели
input_layer = Input(shape=(70,))
embedding_layer = Embedding(
    input_dim=embedding_matrix.shape[0],
    output_dim=embedding_matrix.shape[1],
    weights=[embedding_matrix],
    trainable=False
)(input_layer)
lstm_1 = LSTM(128, return_sequences=True)(embedding_layer)
lstm_2 = LSTM(64)(lstm_1)
dropout = Dropout(0.3)(lstm_2)
output = Dense(7, activation='sigmoid')(dropout)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer='adam', loss=loss_fn, metrics=[F1Score(), Precision(), Recall()])
model.summary()

from keras._tf_keras.keras.callbacks import EarlyStopping, ModelCheckpoint

checkpoint_cb = ModelCheckpoint(
    "sran_model_v3.keras",
    save_best_only=True,
    monitor="val_loss",
    mode="min",
    verbose=1
)

earlystop_cb = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    callbacks=[checkpoint_cb, earlystop_cb],
    verbose=1
)
