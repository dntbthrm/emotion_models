import numpy as np
import pickle
import tensorflow as tf
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences

model1 = tf.keras.models.load_model("emotion_model.h5")
model2 = tf.keras.models.load_model("emotion_model2.h5")
model3 = tf.keras.models.load_model("emotion_model3.h5")
model_small = tf.keras.models.load_model("emotion_model_small.h5")

with open("model_data/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("model_data/tokenizer_small.pkl", "rb") as f:
    tokenizer_small = pickle.load(f)

with open("model_data/label_to_index.pkl", "rb") as f:
    label_classes = pickle.load(f)

with open("model_data/label_to_index_small.pkl", "rb") as f:
    label_classes_small = pickle.load(f)

MAX_SEQUENCE_LENGTH = 100
THRESHOLD = 0.2

while True:
    text = input("Введите текст (или 'exit' для выхода): ")
    if text.lower() == "exit":
        break

    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)

    sequence_small = tokenizer_small.texts_to_sequences([text])
    padded_sequence_small = pad_sequences(sequence_small, maxlen=MAX_SEQUENCE_LENGTH)


    prediction1 = model1.predict(padded_sequence)[0]
    prediction2 = model2.predict(padded_sequence)[0]
    prediction3 = model3.predict(padded_sequence)[0]
    prediction_small = model_small.predict(padded_sequence_small)[0]

    detected_emotions1 = [label_classes[i] for i, prob in enumerate(prediction1) if prob > THRESHOLD]
    detected_emotions2 = [label_classes[i] for i, prob in enumerate(prediction2) if prob > THRESHOLD]
    detected_emotions3 = [label_classes[i] for i, prob in enumerate(prediction3) if prob > THRESHOLD]
    detected_emotions_small = [label_classes_small[i] for i, prob in enumerate(prediction_small) if prob > THRESHOLD]
    print(f"Предсказанные эмоции 1 модели: {detected_emotions1}")
    print(f"Предсказанные эмоции 2 модели: {detected_emotions2}")
    print(f"Предсказанные эмоции 3 модели: {detected_emotions3}")
    print(f"Предсказанные эмоции 4 модели: {detected_emotions_small}")