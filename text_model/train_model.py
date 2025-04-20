import numpy as np
import pickle
import tensorflow as tf
from model import build_model
from keras._tf_keras.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import model_config as mc

#X = np.load("train_test_data/X_train_small.npy")
#y = np.load("train_test_data/y_train_small.npy")
tf.config.set_visible_devices([], 'GPU')
X = np.load("train_test_data/X_train_small.npy")
y = np.load("train_test_data/y_train_small.npy")

# лейблы из файла
with open("v6/label_to_index_small.pkl", "rb") as f:
    label_classes = pickle.load(f)
print(label_classes)

num_classes = len(label_classes)
model = build_model(num_classes)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    filepath='best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

history_new = model.fit(X, y,
                        batch_size=mc.BATCH_SIZE,
                        epochs=15,
                        validation_split=0.15,
                        callbacks=[early_stop],
                        #verbose=1
                        )

np.save("v6/training_history_small_v5.npy", history_new.history)
model.save("./emotion_model_small_v13.keras")
print("Улучшенная модель обучена и сохранена.")