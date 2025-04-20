import pandas as pd

# Конфигурация
BASIC_EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']
INPUT_PATH = "../../dataset/ru-go-emotions-raw.csv"
OUTPUT_PATH = "../../dataset/ru-go-emotions-preprocessed_v8.csv"



# Словарь маппинга: исходная эмоция -> список базовых эмоций
EMOTION_MAPPING = {
    # Сохраняем
    'amusement': ['joy'],
    'grief': ['sadness'],
    'optimism': ['joy'],
    'annoyance': ['anger', 'disgust'],
    'confusion': ['sadness', 'fear'],
    'disappointment': ['sadness', 'surprise'],
    'nervousness': ['fear', 'sadness'],
    'remorse': ['sadness', 'disgust'],
    'embarrassment': ['sadness'],
    'disapproval': ['disgust'],

    # Удаляем
    # 'approval', 'caring', 'desire', 'gratitude',
    # 'love', 'pride', 'relief', 'excitement'

    # Перенаправляем
    #'realization': ['surprise'],  # вместо neutral
    'curiosity': ['surprise']  # вместо neutral
}

# Эмоции, которые будут считаться нейтральными если нет других меток
NON_DEFINED_EMOTIONS = [
    #'approval',
    #'caring',
    #'desire',
    #'gratitude',
    #'love',
    #'realization',
    #'relief',
    'curiosity',
    #'pride'
]

ALL_EMOTIONS = [
'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust',
 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love',
 'nervousness', 'neutral', 'optimism', 'pride', 'realization', 'relief',
 'remorse', 'sadness', 'surprise'
]

IGNORED_EMOTIONS = {
    'approval', 'caring', 'desire', 'gratitude',
    'love', 'pride', 'relief', 'excitement', 'realization'
}


# Полный код с фильтрацией
keep_columns = ['ru_text'] + ALL_EMOTIONS



# Загрузка данных
raw_data = pd.read_csv(INPUT_PATH)[keep_columns]

raw_data = raw_data.drop(columns=IGNORED_EMOTIONS, errors='ignore')

#raw_data = pd.read_csv(INPUT_PATH)
proc_data = pd.DataFrame()
proc_data['ru_text'] = raw_data['ru_text']

print(raw_data.columns)

# Инициализация базовых эмоций нулями
for emotion in BASIC_EMOTIONS:
    proc_data[emotion] = 0

# Шаг 0: Копируем исходный 'neutral'
proc_data['neutral'] = raw_data['neutral'].astype(int)

# Шаг 1: Обработка всех эмоций, кроме non_defined
for emotion, targets in EMOTION_MAPPING.items():
    if emotion in NON_DEFINED_EMOTIONS:
        continue
    if emotion in raw_data.columns:
        mask = raw_data[emotion] == 1
        for target in targets:
            # Убираем перезапись нейтральных, если они уже есть
            proc_data.loc[mask & (proc_data['neutral'] == 0), target] = 1

# Шаг 2: Обработка non-defined эмоций (добавляем условие для нейтральных)
for emotion in NON_DEFINED_EMOTIONS:
    if emotion in raw_data.columns:
        emotion_mask = raw_data[emotion] == 1
        other_emotions_mask = (
                raw_data
                .drop(columns=['ru_text'] + NON_DEFINED_EMOTIONS + ['neutral'])  # Исключаем neutral из проверки
                .sum(axis=1) == 0
        )
        final_mask = emotion_mask & other_emotions_mask
        targets = EMOTION_MAPPING[emotion]

        # Добавляем нейтральные, если нет других меток
        for target in targets:
            proc_data.loc[final_mask & (proc_data['neutral'] == 0), target] = 1

        # Если после всех маппингов нет меток - ставим neutral
        no_labels = final_mask & (proc_data[BASIC_EMOTIONS].sum(axis=1) == 0)
        proc_data.loc[no_labels, 'neutral'] = 1


# Удаление строк без эмоций
final_data = proc_data[proc_data[BASIC_EMOTIONS].sum(axis=1) > 0]

# Сохранение
final_data.to_csv(OUTPUT_PATH, index=False)

# Статистика
print(f"Исходный размер: {raw_data.shape}")
print(f"Финальный размер: {final_data.shape}")
print(f"Удалено строк: {len(raw_data) - len(final_data)}")
print("Колонки:", final_data.columns.tolist())

print(final_data[BASIC_EMOTIONS].sum().sort_values(ascending=False))