import pandas as pd
import numpy as np

whole_emotions = [
'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust',
 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love',
 'nervousness', 'neutral', 'optimism', 'pride', 'realization', 'relief',
 'remorse', 'sadness', 'surprise'
]

basic_emotions = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']

basic_dict = {'anger' : 0, 'disgust': 1, 'fear' : 2, 'joy': 3, 'sadness': 4, 'surprise': 5, 'neutral': 6}

trash = ['text', 'id', 'author', 'subreddit', 'link_id', 'parent_id',
       'created_utc', 'rater_id', 'example_very_unclear']

priority = {
    'fear': 1,
    'anger': 2,
    'joy': 3,
    'sadness': 4,
    'surprise': 5,
    'disgust': 6,
    'neutral': 7
}

emotion_mapping = {
    #'admiration': 'joy',
    #'amusement': 'joy',
    #'approval': 'joy',
    'caring': 'sadness',
    'confusion': 'fear',  # не neutral!
    'curiosity': 'surprise',  # вместо neutral → surprise
    #'desire': 'joy',
    'disappointment': 'sadness',
    'disapproval':  'disgust',  # вместо anger → disgust
    'embarrassment': 'fear', #['sadness', 'fear'],
    'excitement': 'joy',
    #'gratitude': 'joy',
    'grief': 'sadness',
    #'love': 'joy',
    'nervousness': 'fear',
    'optimism': 'joy',
    'pride': 'joy',
    'realization': 'surprise',  # вместо neutral → surprise
    #'relief': 'joy',
    'remorse': 'sadness',
}

raw_data = pd.read_csv("../../dataset/ru-go-emotions-raw.csv")
print(f"real shape: {raw_data.shape}")
print(f"columns: {raw_data.columns}")

raw_data = raw_data.drop(trash, axis = 1)

print(f"new columns: {raw_data.columns}")

old_kolv0 = (raw_data[whole_emotions].sum(axis=1) == 0).sum()

print(f"nulls {old_kolv0}")

# удаление строк без эмоций
raw_data = raw_data[raw_data[whole_emotions].sum(axis=1) != 0]
print(f"not null shape: {raw_data.shape}")

# кол-во эмоций по классам
df = raw_data[whole_emotions].sum().sort_values(ascending=False).reset_index()
print("распределение по классам:\n",df)


for complex_emotion, basic_emotion in emotion_mapping.items():
    if isinstance(basic_emotion, list):  # если это список
        for emotion in basic_emotion:
            raw_data[emotion] += raw_data[complex_emotion]
    elif complex_emotion in raw_data.columns:
        raw_data[basic_emotion] += raw_data[complex_emotion]

for emotion in basic_emotions:
    raw_data[emotion] = raw_data[emotion].clip(upper=1)

columns_to_drop = list(set(whole_emotions) - set(basic_emotions))
raw_data.drop(columns=columns_to_drop, axis=1, inplace=True)

print(raw_data.columns)

df_1 = raw_data[basic_emotions].sum().sort_values(ascending=False).reset_index()
print("распределение по классам basic:\n",df_1)

# Фильтруем строки, где neutral = 1
neutral_rows = raw_data[raw_data['neutral'] == 1]

# Если их больше 23 тыс., выбираем случайно 23 тыс. строк
if len(neutral_rows) > 23000:
    neutral_rows = neutral_rows.sample(n=23000, random_state=42)

# Слияние с оставшимися данными
other_rows = raw_data[raw_data['neutral'] == 0]
raw_data_simplified = pd.concat([neutral_rows, other_rows])

raw_data = raw_data_simplified
# Проверяем результат
#print(raw_data_simplified['neutral'].sum())  # Должно быть 23 тыс.

df_1 = raw_data[basic_emotions].sum().sort_values(ascending=False).reset_index()
print("распределение по классам basic нейтрал дроп:\n",df_1)

old_kolv0 = (raw_data[basic_emotions].sum(axis=1) == 0).sum()

print(f"nulls new {old_kolv0}")

# удаление строк без эмоций
raw_data = raw_data[raw_data[basic_emotions].sum(axis=1) != 0]
print(f"not null shape: {raw_data.shape}")

new_rows = []

# замена эмоций на лейбл + дубли
'''for index, row in raw_data.iterrows():
    present_emotions = [emotion for emotion in basic_emotions if row[emotion] == 1]
    #print(present_emotions)
    for emotion in present_emotions:
        new_row = row.to_dict()
        new_row['label'] = basic_dict.get(emotion)
        new_rows.append(new_row)'''
for index, row in raw_data.iterrows():
    present_emotions = [emotion for emotion in basic_emotions if row[emotion] == 1]

    if len(present_emotions) == 1:
        main_emotion = sorted(present_emotions, key=lambda x: priority[x])[0]
    elif len(present_emotions) > 1:
        main_emotion = sorted(present_emotions)[0]
    else:
        continue  # если нет эмоций, пропускаем

    new_row = row.to_dict()
    new_row['label'] = basic_dict.get(main_emotion)
    new_rows.append(new_row)

new_df = pd.DataFrame(new_rows)

new_df = new_df.drop(basic_emotions,axis=1)

new_df.to_csv('../../dataset/updated_data_1.csv', index=False)
print(new_df.shape)
print(new_df.head())


