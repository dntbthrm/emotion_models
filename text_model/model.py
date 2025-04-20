import tensorflow as tf
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from keras._tf_keras.keras.metrics import Precision, Recall, AUC
from keras._tf_keras.keras.optimizers import AdamW
import model_config as mc

#def build_model(num_classes, vocab_size=20000, embedding_dim=128, max_length=100):
'''def build_model(num_classes, vocab_size=mc.MAX_VOCAB_SIZE, embedding_dim=mc.EMBEDDING_DIM, max_length=mc.MAX_SEQUENCE_LENGTH):
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

    return model'''
def build_model(num_classes, vocab_size=mc.MAX_VOCAB_SIZE, embedding_dim=mc.EMBEDDING_DIM, max_length=mc.MAX_SEQUENCE_LENGTH):
    '''model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        #Bidirectional(LSTM(mc.LSTM_UNITS, return_sequences=True)),
        #Dropout(0.3),
        LSTM(16),
        #Dropout(0.3),
        Dense(16, activation="relu"),
        Dropout(0.2),
        Dense(num_classes, activation="softmax")
    ])'''
    model = Sequential([
        Embedding(input_dim=20000, output_dim=256, input_length=100),
        LSTM(128, return_sequences=True, recurrent_dropout=0.3),
        LSTM(32),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(7, activation='sigmoid')
    ])

    optimizer = AdamW(learning_rate=1e-3, weight_decay=1e-5)

    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=['accuracy', AUC(), Precision(), Recall()]
    )

    return model
