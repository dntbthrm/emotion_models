import pandas as pd
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from transformers import BertTokenizer, BertForSequenceClassification, DataCollatorWithPadding
import torch

basic_dict = {'anger' : 0, 'disgust': 1, 'fear' : 2, 'joy': 3, 'sadness': 4, 'surprise': 5, 'neutral': 6}

df = pd.read_csv("../../dataset/updated_data_cleaned.csv")
print(df.isna().sum())     # NaN по каждому столбцу
print(df.isna().sum().sum())    # Общее количество NaN
print (df.shape)
from collections import Counter
label_counts = dict(Counter(df['label']))
named_counts = {emotion: label_counts.get(code, 0) for emotion, code in basic_dict.items()}
print("\nРаспределение ГОТОВОЕ")
print(named_counts)

from sklearn.utils import shuffle

balanced = []
samples_per_class_train = 2000
samples_per_class_val = 250
samples_per_class_test = 250

for cls in sorted(df['label'].unique()):
    cls_df = df[df['label'] == cls]
    cls_df = shuffle(cls_df, random_state=42)

    train_part = cls_df[:samples_per_class_train]
    val_part = cls_df[samples_per_class_train:samples_per_class_train + samples_per_class_val]
    test_part = cls_df[samples_per_class_train + samples_per_class_val:
                       samples_per_class_train + samples_per_class_val + samples_per_class_test]

    balanced.append((train_part, val_part, test_part))

# Собираем итоговые выборки
train_df = pd.concat([x[0] for x in balanced]).reset_index(drop=True)
val_df = pd.concat([x[1] for x in balanced]).reset_index(drop=True)
test_df = pd.concat([x[2] for x in balanced]).reset_index(drop=True)

# Сохраняем
train_df.to_csv("train_balanced.csv", index=False)
val_df.to_csv("val_balanced.csv", index=False)
test_df.to_csv("test_balanced.csv", index=False)

print("Сохранено сбалансированных: train / val / test")


import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


df = pd.read_csv("../../dataset/updated_data_cleaned2.csv")

# подсчёт количества каждого класса
label_counts = df['label'].value_counts().sort_index()

# гистограммы с подписями классов
plt.figure(figsize=(8, 5))
plt.bar(basic_dict.keys(), [label_counts[i] for i in basic_dict.values()], color='skyblue')
plt.xlabel("Эмоции")
plt.ylabel("Количество")
plt.title("Распределение классов эмоций в датасете")
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)

#plt.savefig("emotion_distribution.png")

# ------- ТЕСТ ----------
from torch.utils.data import Dataset, DataLoader

target_names = [k for k, v in sorted(basic_dict.items(), key=lambda item: item[1])]

device = torch.device('cpu')
model_dir = "emotion_bert_best"
#model_dir = "saved_rubert_model"
model = BertForSequenceClassification.from_pretrained(model_dir)
tokenizer = BertTokenizer.from_pretrained(model_dir)
model.eval()

test_df = pd.read_csv('test.csv')
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

test_dataset = EmotionDataset(test_df['ru_text'], test_df['label'], tokenizer)
test_loader = DataLoader(test_dataset, batch_size=10, collate_fn=data_collator)
y_pred, true_labels, y_pred_proba = [], [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Тест"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

        logits = outputs.logits
        probs = F.softmax(logits, dim=1)

        preds = torch.argmax(probs, dim=1)

        y_pred.extend(preds.cpu().numpy())
        true_labels.extend(batch['labels'].cpu().numpy())
        y_pred_proba.extend(probs.cpu().numpy())

print("\nРезультаты на тестовой выборке:")
print(classification_report(true_labels, y_pred, digits=4, target_names=target_names))


import numpy as np
from sklearn.metrics import classification_report, f1_score

# Исходные метки
y_true = np.array(true_labels)
num_classes = len(set(y_true))
label_mapping = {'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3, 'sadness': 4, 'surprise': 5, 'neutral': 6}
inv_label_mapping = {v: k for k, v in label_mapping.items()}

y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
roc_auc_scores = roc_auc_score(y_true_bin, y_pred_proba, average=None)

print("\nROC AUC по классам:")
for i, score in enumerate(roc_auc_scores):
    print(f"{inv_label_mapping[i]}: {score:.4f}")

# classification report
target_names = [inv_label_mapping[i] for i in range(7)]
report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
print(report)


import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score


n_classes = y_pred_proba.shape[1]

y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

precision = dict()
recall = dict()
average_precision = dict()

for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
    average_precision[i] = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])

# Взвешенная AP
average_precision["weighted"] = average_precision_score(y_true_bin, y_pred_proba, average="weighted")

from sklearn.metrics import roc_curve, auc
import torch.nn.functional as F

num_classes = 7
target_names = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']

# получение вероятностей
probs, true_labels = [], []

model.eval()
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Тест"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        softmaxed = F.softmax(outputs.logits, dim=1)

        probs.append(softmaxed.cpu().numpy())
        true_labels.extend(batch['labels'].cpu().numpy())

probs = np.concatenate(probs, axis=0)
true_labels = np.array(true_labels)

y_true_bin = label_binarize(true_labels, classes=range(num_classes))

plt.figure(figsize=(10, 8))
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{target_names[i]} (AUC = {roc_auc:.4f})")

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривые по классам')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_auc_multiclass.png")
