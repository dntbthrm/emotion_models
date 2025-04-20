import numpy as np
import pandas as pd
from ast import literal_eval

# Загрузим как строки (если уже не загружено)
df = pd.read_csv("v10_val.csv")

# Преобразуем каждую строку-метку в список целых чисел
df['label'] = df['label'].apply(lambda x: np.array(list(map(int, x.strip('[]').split()))))

# Преобразуем в массив, чтобы можно было суммировать по колонкам
labels_matrix = np.vstack(df['label'].values)  # (num_samples, num_classes)

# Суммируем по каждой колонке (по каждому классу)
class_sums = labels_matrix.sum(axis=0)

# Привяжем к названиям эмоций
BASIC_EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']
emotion_counts = pd.Series(class_sums, index=BASIC_EMOTIONS)

# Отсортируем по убыванию
print(emotion_counts.sort_values(ascending=False))
