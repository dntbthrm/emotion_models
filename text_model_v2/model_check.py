import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    ConfusionMatrixDisplay
)

# классы
target_names = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']

test_df = pd.read_csv("test.csv")
y_true = test_df['label'].values
y_pred = np.load("y_pred_magic.npy")

# метрики
report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred)

# матрица ошибок
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
plt.title("Матрица ошибок")
plt.xlabel("Предсказано")
plt.ylabel("Истинное")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
#plt.show()

# графики Precision, Recall, F1 по каждому классу
metrics = {
    "Precision": precision,
    "Recall": recall,
    "F1-score": f1
}

for metric_name, metric_values in metrics.items():
    plt.figure(figsize=(8, 4))
    sns.barplot(x=target_names, y=metric_values)
    plt.ylim(0, 1)
    plt.title(f"{metric_name} по классам")
    plt.ylabel(metric_name)
    plt.xlabel("Класс")
    plt.tight_layout()
    plt.savefig(f"{metric_name.lower()}_per_class.png")
    #plt.show()

# круговая диаграмма предсказанных классов
plt.figure(figsize=(6, 6))
unique, counts = np.unique(y_pred, return_counts=True)
plt.pie(counts, labels=[target_names[i] for i in unique], autopct='%1.1f%%', startangle=140)
plt.title("Распределение предсказанных классов")
plt.tight_layout()
plt.savefig("predicted_distribution_pie.png")
#plt.show()

# вывод отчета
#report_df = pd.DataFrame(report).transpose()
print("\nТаблица метрик по классам:")
print(classification_report(y_true, y_pred, target_names=target_names, digits=2))
with open("classification_report.txt", "w", encoding="utf-8") as f:
    f.write(classification_report(y_true, y_pred, target_names=target_names, digits=2))
