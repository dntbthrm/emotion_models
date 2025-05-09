import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", font_scale=1.1)

ravdess = pd.read_csv("ravdess_features_2.csv")
crema = pd.read_csv("crema_features_2.csv")
tess = pd.read_csv("tess_features_2.csv")
# в crema нет удивления, его не учитываем
data = pd.concat([ravdess, tess], ignore_index=True)

# Добавление информации о происхождении
ravdess['dataset'] = 'RAVDESS'
crema['dataset'] = 'CREMA'
tess['dataset'] = 'TESS'
data['dataset'] = 'Combined'

# Словарь кодов эмоций
emotion_codes = {
    "neutral": 1,
    "happy": 2,
    "sad": 3,
    "angry": 4,
    "fearful": 5,
    "disgust": 6,
    "surprised": 7
}
inv_emotion_codes = {v: k for k, v in emotion_codes.items()}


# Функция построения графика распределения
def plot_emotion_distribution(df, name, save=True):
    counts = df['label'].value_counts().sort_index()
    counts_named = counts.rename(index=inv_emotion_codes)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=counts_named.index, y=counts_named.values, palette="muted")
    plt.title(f"Распределение эмоций — {name}")
    plt.xlabel("Эмоция")
    plt.ylabel("Количество")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save:
        plt.savefig(f"emotion_distribution_{name.lower()}.png")
    plt.close()


# Построение графиков
plot_emotion_distribution(ravdess, "RAVDESS")
plot_emotion_distribution(crema, "CREMA")
plot_emotion_distribution(tess, "TESS")
plot_emotion_distribution(data, "Combined")

# Сводная таблица распределения
summary = data.groupby(['dataset', 'label']).size().unstack(fill_value=0)
summary_named = summary.rename(columns=inv_emotion_codes)

# Вывод сводной таблицы
print(summary_named)


# Функция построения pie chart
def plot_emotion_pie(df, name, save=True):
    counts = df['label'].value_counts().sort_index()
    counts_named = counts.rename(index=inv_emotion_codes)

    plt.figure(figsize=(7, 7))
    plt.pie(
        counts_named.values,
        labels=counts_named.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=sns.color_palette("pastel")
    )
    plt.title(f"Доля эмоций — {name}")

    if save:
        plt.savefig(f"emotion_pie_{name.lower()}.png")
    plt.close()


# Построение pie charts
plot_emotion_pie(ravdess, "RAVDESS")
plot_emotion_pie(crema, "CREMA")
plot_emotion_pie(tess, "TESS")
plot_emotion_pie(data, "Combined")
