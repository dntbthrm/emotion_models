import pandas as pd
from sklearn.model_selection import train_test_split

# Загрузим датафрейм (предположим, что он уже существует)
df = pd.read_csv('../../dataset/updated_data.csv')

# Разделим на обучающую, валидационную и тестовую выборки
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)  # 80% обучающая выборка
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)  # 10% валидация, 10% тест

# Посмотрим на размеры выборок
print(f"Train size: {len(train_df)}")
print(f"Validation size: {len(val_df)}")
print(f"Test size: {len(test_df)}")

# Сохраняем выборки в csv файлы
train_df.to_csv('../../dataset/train_data.csv', index=False)
val_df.to_csv('../../dataset/val_data.csv', index=False)
test_df.to_csv('../../dataset/test_data.csv', index=False)

import numpy as np
from transformers import BertTokenizer

# Инициализация токенизатора для BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Токенизация данных
train_encodings = tokenizer(train_df['ru_text'].tolist(), truncation=True, padding=True)
val_encodings = tokenizer(val_df['ru_text'].tolist(), truncation=True, padding=True)
test_encodings = tokenizer(test_df['ru_text'].tolist(), truncation=True, padding=True)
tokenizer.save_pretrained('./model_save')  # Сохранение токенизатора


# Функция для сохранения данных в NumPy
def save_to_numpy(data, filename):
    # Преобразуем BatchEncoding в NumPy массивы
    input_ids = np.array(data['input_ids'])
    attention_mask = np.array(data['attention_mask'])

    # Сохраняем массивы в .npy файлы
    np.save(f"{filename}_input_ids.npy", input_ids)
    np.save(f"{filename}_attention_mask.npy", attention_mask)


# Сохраним токенизированные данные в NumPy файлы
save_to_numpy(train_encodings, 'trans_data/train_encodings')
save_to_numpy(val_encodings, 'trans_data/val_encodings')
save_to_numpy(test_encodings, 'trans_data/test_encodings')

print("Токенизированные данные сохранены в формате .npy.")
