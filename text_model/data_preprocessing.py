import pandas as pd
import numpy as np

''' 
Изменение оригинального датасета в соответствии с 
базовыми эмоциями
'''

whole_emotions = [
'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust',
 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love',
 'nervousness', 'neutral', 'optimism', 'pride', 'realization', 'relief',
 'remorse', 'sadness', 'surprise'
]

basic_emotions = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']
defined_switch = {
    #'admiration': {'joy'}, # восхищение
    'amusement': {'joy'}, # увлечение
    'annoyance': {'anger', 'disgust'},
    'confusion': {'sadness', 'fear'},
    #'curiosity': {'surprise'},
    'disappointment': {'sadness', 'surprise'},
   # 'disapproval': {'disgust', 'anger'},
   # 'embarrassment': {'sadness', 'fear'},
   # 'excitement': {'surprise', 'joy'},
    'grief': {'sadness'},
    'nervousness': {'fear', 'sadness'},
    'optimism': {'joy'},
    #'pride': {'joy'},
    'remorse': {'sadness', 'disgust'}
}

emotion_mapping = {
    'admiration': 'joy',
    'amusement': 'joy',
    #'anger': 'anger',
    'annoyance': 'anger',
    'approval': 'joy',
    'caring': 'joy',
    'confusion': 'neutral',
    'curiosity': 'neutral',
    'desire': 'joy',
    'disappointment': 'sadness',
    'disapproval': 'disgust',
    #'disgust': 'disgust',
    'embarrassment': 'sadness',
    'excitement': 'joy',
    #'fear': 'fear',
    'gratitude': 'joy',
    'grief': 'sadness',
    #'joy': 'joy',
    'love': 'joy',
    'nervousness': 'fear',
    #'neutral': 'neutral',
    'optimism': 'joy',
    'pride': 'joy',
    'realization': 'neutral',
    'relief': 'joy',
    'remorse': 'sadness',
    #'sadness': 'sadness',
    #'surprise': 'surprise'
}

non_defined = {'approval', 'caring', 'desire', 'gratitude', 'love', 'realization', 'relief', 'curiosity', 'pride'}

raw_data = pd.read_csv("../../dataset/ru-go-emotions-raw.csv")

proc_data = raw_data.copy()


# изменение классов, где эмоция определяется комбинацией базовых
def change_def_emotion(index, column):
    #basic_list = defined_switch[column]
    basic_list = emotion_mapping[column]
    for emo in basic_list:
        proc_data.at[index, emo] = 1

def change_non_def(index):
    is_neutral = 1
    for emo in whole_emotions:
        if proc_data.at[index, emo] == 1 and emo != 'neutral':
            is_neutral = 0
            break
    if is_neutral == 0:
        proc_data.at[index, 'neutral'] = 1

print("Оригинальный датасет: ", raw_data.shape, "\nКолонки: ", raw_data.columns)


for i in range (proc_data.shape[0]):
    for col in proc_data.columns:
        if col in defined_switch and proc_data.at[i, col] == 1:
            change_def_emotion(i, col)


norm_columns = ['ru_text'] + basic_emotions

proc_data = proc_data[norm_columns].dropna()
#count_unclear = raw_data[raw_data["example_very_unclear"] == False].shape[0]
#print(f"Количество строк, где example_very_unclear == True: {count_unclear}")

nulls_proc = (proc_data[basic_emotions].sum(axis=1) == 0)
kolv0 = (proc_data[basic_emotions].sum(axis=1) == 0).sum()
nulls_raw = (raw_data[basic_emotions].sum(axis=1) == 0)
old_kolv0 = (raw_data[basic_emotions].sum(axis=1) == 0).sum()
print(f"Novy dataset: {kolv0}; Stary dataset: {old_kolv0}")
proc_data.to_csv("../../dataset/ru-go-emotions-preprocessed_v7.csv", index=False)
print("Новый датасет: ", proc_data.shape, "\nКолонки: ", proc_data.columns, "\nФайл csv сохранен вне папки")
print("Новый датасет: ", proc_data.shape)

#print(proc_data[nulls_proc].iloc[0])
#print(raw_data[nulls_raw].iloc[0])

proc_data = proc_data[~nulls_proc].reset_index(drop=True)
print("Новый датасет 2: ", proc_data.shape)

import matplotlib.pyplot as plt
import seaborn as sns
emotion_counts = proc_data[basic_emotions].sum().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(
    x=emotion_counts.index,
    y=emotion_counts.values,
    hue=emotion_counts.index,
    palette="mako",
    legend=False
)
plt.title("Частота эмоций (multi-label)", fontsize=16)
plt.xlabel("Эмоции")
plt.ylabel("Количество вхождений")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("images/proc_data_class_v7.png")

emotion_counts = raw_data[whole_emotions].sum().sort_values(ascending=False)

plt.figure(figsize=(24, 12))
sns.barplot(
    x=emotion_counts.index,
    y=emotion_counts.values,
    hue=emotion_counts.index,
    palette="mako",
    legend=False
)
plt.title("Частота эмоций (multi-label)", fontsize=16)
plt.xlabel("Эмоции")
plt.ylabel("Количество вхождений")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("images/raw_data_class_v7.png")





