import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Загружаем тестовые данные
#X_test = np.load("model_data/X_test_small.npy")
#y_test = np.load("model_data/y_test_small.npy")

X_test = np.load("train_test_data/X_test_small.npy")
y_test = np.load("train_test_data/y_test_small.npy")

# Загружаем модель
#model = tf.keras.models.load_model("emotion_model_small.h5")
model = tf.keras.models.load_model("best_model.keras")
# Загружаем метки классов
#with open("train_test_data/label_to_index_small.pkl", "rb") as f:
with open("model_data/label_to_index_small.pkl", "rb") as f:
    label_classes = pickle.load(f)

print(label_classes)
# Обратный словарь {индекс: эмоция}
index_to_label = {idx: label for idx, label in enumerate(label_classes)}


# Предсказание на тестовом наборе
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)  # Преобразуем вероятности в метки классов


y_true = np.argmax(y_test, axis=1)  # Истинные метки классов
#y_true = y_test
print("Уникальные метки в y_true:", np.unique(y_true))
print("Уникальные метки в y_pred:", np.unique(y_pred))

# Вывод отчёта о качестве модели
report = classification_report(
    y_true,
    y_pred,
    target_names=[index_to_label[i] for i in range(len(index_to_label))],
    labels=np.arange(len(index_to_label))  # Все классы от 0 до N-1
)


print("🔍 Оценка модели:\n")
print(report)

# ======= Построение графиков =======

# 1️⃣ График изменения точности (accuracy) и потерь (loss) во время обучения
history = np.load("data_arrays/training_history_small_v3.npy", allow_pickle=True).item()

plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history["accuracy"], label="Train Accuracy")
plt.plot(history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("График точности модели")
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history["loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("График функции потерь")
plt.legend()

#plt.savefig("images/accuracy_loss_small_v3.png")

# 2️⃣ Матрица ошибок (confusion matrix)
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=[index_to_label[i] for i in range(len(index_to_label))],
            yticklabels=[index_to_label[i] for i in range(len(index_to_label))])
plt.xlabel("Предсказанные классы")
plt.ylabel("Истинные классы")
plt.title("Матрица ошибок (Confusion Matrix)")
#plt.savefig("images/confusion_matrix_small_v3.png")
