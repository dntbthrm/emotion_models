import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score

emotion_codes = {
    1: "neutral",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fearful",
    6: "disgust",
    7: "surprised"
}

model = joblib.load("audio_classifier_big.pkl")

data = pd.read_csv("test_dataset_v1.csv")
X_test = data.drop(columns=["label"])
y_test = data["label"]

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
report = classification_report(y_test, y_pred, target_names=[emotion_codes[i] for i in range(1, 8)])

with open("test_report_big.txt", "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"F1-score: {f1:.4f}\n")
    f.write(report)

cm = confusion_matrix(y_test, y_pred, labels=list(emotion_codes.keys()))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[emotion_codes[i] for i in range(1, 8)])

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix (Test)")
plt.tight_layout()
plt.savefig("confusion_matrix_test_big.png")

from sklearn.metrics import precision_recall_fscore_support

_, _, f1_per_class, _ = precision_recall_fscore_support(y_test, y_pred, labels=list(emotion_codes.keys()))

plt.figure(figsize=(10, 5))
sns.barplot(x=[emotion_codes[i] for i in range(1, 8)], y=f1_per_class)
plt.title("F1-score по каждому классу (Test)")
plt.ylabel("F1-score")
plt.xlabel("Класс эмоции")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("f1_per_class_test_big.png")


N = 7

df_plot = X_test.copy()
df_plot["label"] = y_test.map(emotion_codes)

fig, axes = plt.subplots(3, 3, figsize=(18, 10))
axes = axes.flatten()


for i, feature in enumerate(X_test.columns):
    sns.boxplot(x="label", y=feature, data=df_plot, ax=axes[i])
    axes[i].set_title(f"{feature} по эмоциям")
    axes[i].set_xlabel("Эмоция")
    axes[i].set_ylabel(feature)
    axes[i].tick_params(axis='x', rotation=45)

for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig("boxplots_big.png")
