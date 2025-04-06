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
defined_switch = {
    'admiration': {'joy'},
    'amusement': {'joy'},
    'annoyance': {'anger'},
    'confusion': {'sadness', 'surprise'},
    'curiosity': {'surprise'},
    'disappointment': {'sadness', 'surprise'},
    'disapproval': {'surprise', 'anger'},
    'embarrassment': {'sadness', 'surprise'},
    'excitement': {'fear', 'joy'},
    'grief': {'sadness'},
    'nervousness': {'fear', 'sadness'},
    'optimism': {'joy'},
    'pride': {'joy'},
    'remorse': {'sadness', 'disgust'}
}

non_defined = {'approval', 'caring', 'desire', 'gratitude', 'love', 'realization', 'relief'}

raw_data = pd.read_csv("../../dataset/ru-go-emotions-raw.csv")

proc_data = raw_data.copy()

def change_def_emotion(index, column):
    basic_list = defined_switch[column]
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

for i, raw in enumerate(proc_data):
    for col in proc_data.columns:
        '''if col in non_defined and proc_data.at[i, col] == 1:
            change_non_def(i)'''
        if col in defined_switch and proc_data.at[i, col] == 1:
            change_def_emotion(i, col)

norm_columns = ['ru_text'] + basic_emotions

proc_data = proc_data[norm_columns].dropna()
proc_data.to_csv("../../dataset/ru-go-emotions-preprocessed.csv", index=False)
print("Новый датасет: ", proc_data.shape, "\nКолонки: ", proc_data.columns, "\nФайл csv сохранен вне папки")

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
plt.savefig("images/proc_data_class.png")

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
plt.savefig("images/raw_data_class.png")





