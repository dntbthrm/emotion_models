import tensorflow as tf
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Embedding, LSTM, Dense
import model_config as mc

#def build_model(num_classes, vocab_size=20000, embedding_dim=128, max_length=100):
def build_model(num_classes, vocab_size=mc.MAX_VOCAB_SIZE, embedding_dim=mc.EMBEDDING_DIM, max_length=mc.MAX_SEQUENCE_LENGTH):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        LSTM(mc.LSTM_UNITS, return_sequences=True), #64
        LSTM(32), # u novoy 128
        Dense(32, activation="relu"),
        Dense(num_classes, activation="sigmoid")  # Sigmoid
    ])

    model.compile(
        loss="binary_crossentropy",  # мультиклассоая классификация
        optimizer="adam",
        metrics=["accuracy"]
    )

    return model
