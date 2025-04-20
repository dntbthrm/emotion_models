from keras._tf_keras.keras.layers import Input, Embedding, LSTM, Dropout, Dense, Bidirectional
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.metrics import Precision, Recall, AUC, Metric
from keras._tf_keras.keras import backend as K
import tensorflow as tf


#BEST
tf.config.set_visible_devices([], 'GPU')
input_layer = Input(shape=(70,))
embedding_layer = Embedding(input_dim=20000, output_dim=256)(input_layer)
lstm_1 = Bidirectional(LSTM(128, return_sequences=True))(embedding_layer)
lstm_2 = Bidirectional(LSTM(64))(lstm_1)
dropout = Dropout(0.3)(lstm_2)
output = Dense(7, activation='sigmoid')(dropout)

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

import numpy as np

X_train = np.load("sran_X_train.npy")
X_val = np.load("sran_X_val.npy")
X_test = np.load("sran_X_test.npy")
y_train = np.load("sran_y_train.npy")
y_val = np.load("sran_y_val.npy")
y_test = np.load("sran_y_test.npy")

class_totals = np.sum(y_train, axis=0)
total_labels = np.sum(class_totals)
class_weights = total_labels / (len(class_totals) * class_totals)


def weighted_binary_crossentropy(weights):
    weights = tf.constant(weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        # Клипуем предсказания, чтобы избежать log(0)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        bce = - (weights * y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))
        return K.mean(bce, axis=-1)

    return loss


loss_fn = weighted_binary_crossentropy(class_weights)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer='adam', loss=loss_fn, metrics=[F1Score(), Precision(), Recall()])
model.summary()

from keras._tf_keras.keras.callbacks import EarlyStopping, ModelCheckpoint



# Callbacks
checkpoint_cb = ModelCheckpoint(
    "sran_model_v2_1.keras",
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

# Обучение модели
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=20, 
    batch_size=32,
    callbacks=[checkpoint_cb],
    verbose=1
)