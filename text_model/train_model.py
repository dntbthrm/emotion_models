import numpy as np
import pickle
import tensorflow as tf
from model import build_model
from keras._tf_keras.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import model_config as mc

X = np.load("train_test_data/X_train_small.npy")
y = np.load("train_test_data/y_train_small.npy")

# лейблы из файла
with open("train_test_data/label_to_index_small.pkl", "rb") as f:
    label_classes = pickle.load(f)
print(label_classes)

num_classes = len(label_classes)
model = build_model(num_classes)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,                 # остановка после 5 эпох без улучшений
    restore_best_weights=True  # восстановить лучшие веса
)

checkpoint = ModelCheckpoint(
    filepath='best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,    # уменьшить learning rate в 2 раза
    patience=3,    # если 3 эпохи без улучшения
    min_lr=1e-6,
    verbose=1
)

history_new = model.fit(X, y,
                        batch_size=mc.BATCH_SIZE,
                        epochs=mc.EPOCHS,
                        validation_split=0.2,
                        callbacks=[early_stop, checkpoint, reduce_lr],
                        verbose=1
                        )
np.save("data_arrays/training_history_small_v3.npy", history_new.history)

model.save("./emotion_model_small_v3.keras")
print("Модель обучена и сохранена")
