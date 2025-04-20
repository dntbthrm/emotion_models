import pandas as pd

# Запуск обучения
'''if __name__ == "__main__":
    tf.config.set_visible_devices([], 'GPU')
    classifier = EmotionClassifier(CONFIG)
    history = classifier.train("../../dataset/ru-go-emotions-balanced_min.csv")

    # Сохранение модели и токенайзера
    classifier.model.save("emotion_classifier.keras")
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(classifier.tokenizer, f)'''
BASIC_EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']
ALL_EMOTIONS = [
'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust',
 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love',
 'nervousness', 'neutral', 'optimism', 'pride', 'realization', 'relief',
 'remorse', 'sadness', 'surprise'
]

IGNORED_EMOTIONS = [
    'approval', 'caring', 'desire', 'gratitude',
    'love', 'pride', 'relief', 'excitement', 'realization', 'curiosity'
]

# ненужные столбцы
TO_DROP = ['text', 'id', 'author', 'subreddit', 'link_id', 'parent_id',
       'created_utc', 'rater_id', 'example_very_unclear', 'approval', 'caring', 'desire', 'gratitude',
    'love', 'pride', 'relief', 'excitement', 'realization', 'curiosity']


BAS_IGN = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral', 'approval', 'caring', 'desire', 'gratitude',
    'love', 'pride', 'relief', 'excitement', 'realization', 'curiosity']

df = pd.read_csv('../../dataset/ru-go-emotions-raw.csv')
old_kolv0 = (df[ALL_EMOTIONS].sum(axis=1) == 0).sum()
new_kolv0 = (df[BASIC_EMOTIONS].sum(axis=1) == 0).sum()

new_kolv1 = (df[BAS_IGN].sum(axis=1) == 0).sum()

print(len(ALL_EMOTIONS))
print(df.columns)
print(df.shape)
print((df["example_very_unclear"] == False).sum())
print(f"NULLS ALL: {old_kolv0} NULLS NEW: {new_kolv0} IGNORED: {new_kolv1}")
print(df[ALL_EMOTIONS].sum().sort_values(ascending=False))
from collections import Counter
print('TRAIN--------------------------------------------')
df_train = pd.read_csv('../../dataset/ru-go-emotions-simplified-train.csv')
print(df_train.shape)
df_train['lab'] =  df_train['labels'].apply(lambda x: list(map(int, x.strip("[]").split())))
#print(df_train['lab'].head(30))
print(Counter(sum(df_train['lab'], [])))

print('TEST--------------------------------------------')
df_test = pd.read_csv('../../dataset/ru-go-emotions-simplified-test.csv')
print(df_test.shape)
df_test['lab'] =  df_test['labels'].apply(lambda x: list(map(int, x.strip("[]").split())))
#print(df_train['lab'].head(30))
print(Counter(sum(df_test['lab'], [])))


print('VAL--------------------------------------------')
df_val = pd.read_csv('../../dataset/ru-go-emotions-simplified-validation.csv')
print(df_val.shape)
df_val['lab'] =  df_train['labels'].apply(lambda x: list(map(int, x.strip("[]").split())))
#print(df_train['lab'].head(30))
print(Counter(sum(df_val['lab'], [])))

print(df_train.shape[0] + df_test.shape[0] + df_val.shape[0])



print("DROP--------------------------------------")
df= df.drop(columns=TO_DROP, errors='ignore')
emo_clos = df.drop(columns=['ru_text'])
print(f"after {df.shape}, emptys: {(df[emo_clos.columns].sum(axis=1) == 0).sum()}")

print("DROP EMPTY-------------------------------")
df = df[df[emo_clos.columns].sum(axis=1) > 0]
print(f"Shape {df.shape}, cols: {df.columns}")

EMOTION_MAPPING = {
    'admiration': ['joy'],
    'amusement': ['joy'],
    'grief': ['sadness'],
    'optimism': ['joy'],
    'annoyance': ['anger', 'disgust'],
    'confusion': ['sadness', 'fear'],
    'disappointment': ['sadness', 'surprise'],
    'nervousness': ['fear', 'sadness'],
    'remorse': ['sadness', 'disgust'],
    'embarrassment': ['sadness'],
    'disapproval': ['disgust']
}

TO_STAY = ['ru_text', 'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']

print("CHANGE-----------------------------------")
print(f"emptys: {(df[emo_clos.columns].sum(axis=1) == 0).sum()}")
for emotion, targets in EMOTION_MAPPING.items():
    if emotion in df.columns:
        mask = df[emotion] == 1
        for target in targets:
            # Убираем перезапись нейтральных, если они уже есть
            df.loc[mask, target] = 1
df = df[TO_STAY]
print(f"emptys_2: {(df[BASIC_EMOTIONS].sum(axis=1) == 0).sum()}")
print(df.columns)
print(df[BASIC_EMOTIONS].sum().sort_values(ascending=False))

df.to_csv('../../dataset/ru-go-emotions-v404.csv', index=False)





