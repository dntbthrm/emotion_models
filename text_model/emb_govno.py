import numpy as np
from tqdm import tqdm
from keras._tf_keras.keras.preprocessing.text import tokenizer_from_json
import pickle

embedding_dim = 300
max_words = 20000

# Загрузка токенизатора, если он у тебя уже был сохранен
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Загрузка fastText русских эмбеддингов
embeddings_index = {}
with open('../../dataset/cc.ru.300.vec', encoding='utf-8', newline='\n', errors='ignore') as f:
    next(f)  # Пропускаем первую строку
    for line in tqdm(f, desc="Загрузка fastText в память"):
        values = line.rstrip().split(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print(f"Найдено {len(embeddings_index)} слов с эмбеддингами.")

# Построение embedding matrix
word_index = tokenizer.word_index
num_words = min(max_words, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embedding_dim))

for word, i in word_index.items():
    if i >= max_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

np.save("embedding_matrix.npy", embedding_matrix)
