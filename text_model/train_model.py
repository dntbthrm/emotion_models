import numpy as np
import pickle
import tensorflow as tf
from model import build_model
import model_config as mc

X = np.load("model_data/X.npy")
y = np.load("model_data/y.npy")

# лейблы из файла
with open("model_data/label_to_index.pkl", "rb") as f:
    label_classes = pickle.load(f)

num_classes = len(label_classes)
model = build_model(num_classes)

history_new = model.fit(X, y, batch_size=mc.BATCH_SIZE, epochs=mc.EPOCHS, validation_split=0.2)
np.save("model_data/training_history.npy", history_new.history)

model.save("./emotion_model3.h5")
print("Модель обучена и сохранена")
