'''# Импорт
import pandas as pd
import torch
from sklearn.utils import shuffle
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# ✅ Загрузка и балансировка датасета
df = pd.read_csv("../../dataset/updated_data_cleaned2.csv")  # ← Заменить на имя твоего CSV-файла
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("📡 Устройство:", device)

samples_train = 2000
samples_val = 250
samples_test = 250

balanced_train, balanced_val, balanced_test = [], [], []

for label in sorted(df['label'].unique()):
    class_subset = shuffle(df[df['label'] == label], random_state=42)
    balanced_train.append(class_subset[:samples_train])
    balanced_val.append(class_subset[samples_train:samples_train + samples_val])
    balanced_test.append(class_subset[samples_train + samples_val:samples_train + samples_val + samples_test])

train_df = pd.concat(balanced_train).reset_index(drop=True)
val_df = pd.concat(balanced_val).reset_index(drop=True)
test_df = pd.concat(balanced_test).reset_index(drop=True)

train_df.to_csv('train_balanced.csv', index=False)
val_df.to_csv('val_balanced.csv', index=False)
test_df.to_csv('test_balanced.csv', index=False)

print("✅ Балансировка завершена и сохранена")

# ✅ Tokenizer и модель
model_name = "DeepPavlov/rubert-base-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=7)

# ✅ Токенизация
def tokenize(batch):
    return tokenizer(batch['ru_text'], padding='max_length', truncation=True, max_length=128)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# Удалим лишние поля
columns_to_remove = [col for col in train_dataset.column_names if col not in ['input_ids', 'attention_mask', 'label']]
train_dataset = train_dataset.remove_columns(columns_to_remove)
val_dataset = val_dataset.remove_columns(columns_to_remove)
test_dataset = test_dataset.remove_columns(columns_to_remove)

# ✅ Метрики
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1_macro': f1_score(labels, preds, average='macro'),
        'f1_weighted': f1_score(labels, preds, average='weighted')
    }

# ✅ Тренировочные аргументы
training_args = TrainingArguments(
    #no_cuda=True,
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=13,
    per_device_eval_batch_size=13,
    eval_strategy='epoch',
    save_strategy='epoch',
    logging_dir='./logs',
    load_best_model_at_end=True,
    metric_for_best_model='f1_macro',
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# ✅ Обучение
trainer.train()
# ✅ Сохраняем модель и токенизатор
model_dir = "saved_rubert_model"
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)

print(f"✅ Модель и токенизатор сохранены в: {model_dir}")


# ✅ Оценка на тесте
metrics = trainer.evaluate(test_dataset)
print(f"📊 Accuracy: {metrics['eval_accuracy']:.4f}, F1-macro: {metrics['eval_f1_macro']:.4f}")'''

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, DataCollatorWithPadding
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import numpy as np
import os
import gc


# === 0. Устройство ===
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print("📡 Устройство:", device)
torch.cuda.empty_cache()
gc.collect()
# === 1. Загрузка и разделение датасета ===
df = pd.read_csv("../../dataset/updated_data_cleaned2.csv")  # clean_text, label

train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

train_df.to_csv("train1.csv", index=False)
val_df.to_csv("val1.csv", index=False)
test_df.to_csv("test1.csv", index=False)

print(f"📁 Сохранено: train1.csv ({len(train_df)}), val1.csv ({len(val_df)}), test1.csv ({len(test_df)})")

# === 2. Токенизатор и Dataset ===
tokenizer = BertTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")

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

train_dataset = EmotionDataset(train_df['ru_text'], train_df['label'], tokenizer)
val_dataset = EmotionDataset(val_df['ru_text'], val_df['label'], tokenizer)
test_dataset = EmotionDataset(test_df['ru_text'], test_df['label'], tokenizer)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, collate_fn=data_collator)
val_loader = DataLoader(val_dataset, batch_size=10, collate_fn=data_collator)
test_loader = DataLoader(test_dataset, batch_size=10, collate_fn=data_collator)

# === 3. Классовые веса ===
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df['label']),
    y=train_df['label']
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print(f"⚖️  Веса классов: {class_weights}")

# === 4. Модель, оптимизатор, лосс, планировщик ===
model = BertForSequenceClassification.from_pretrained("DeepPavlov/rubert-base-cased", num_labels=7)
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = CrossEntropyLoss(weight=class_weights)

from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, verbose=True)

# === 5. Тренировка ===
epochs = 10
best_val_f1 = 0
early_stop_counter = 0
early_stop_patience = 3

for epoch in range(epochs):
    print(f"\n🔥 Эпоха {epoch + 1}/{epochs}")
    model.train()
    total_loss = 0

    for step, batch in enumerate(tqdm(train_loader, desc="🔁 Обучение")):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = loss_fn(outputs.logits, batch['labels'])
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (step + 1) % 200 == 0:
            print(f"  🔸 [{step+1}/{len(train_loader)}] Лосс: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"📉 Средний лосс за эпоху: {avg_loss:.4f}")

    # === 6. Валидация ===
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="📐 Валидация"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch['labels'].cpu().numpy())

    val_f1 = f1_score(true_labels, predictions, average='weighted')
    print(f"\n📊 Val F1-score: {val_f1:.4f}")
    print(classification_report(true_labels, predictions, digits=4))

    scheduler.step(val_f1)

    # === 7. Early stopping ===
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        early_stop_counter = 0
        model.save_pretrained("emotion_bert_best")
        tokenizer.save_pretrained("emotion_bert_best")
        print("сохранение лучшей модели")
    else:
        early_stop_counter += 1
        print(f"улучшений нет: {early_stop_counter}/{early_stop_patience}")

    if early_stop_counter >= early_stop_patience:
        print("ОСТАНОВКА")
        break
    '''torch.cuda.empty_cache()
    gc.collect()'''

# === 8. Финальное тестирование ===
print("\n🧪 Финальное тестирование на test.csv")
model = BertForSequenceClassification.from_pretrained("emotion_bert_best").to(device)
model.eval()
predictions, true_labels = [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="🧪 Тест"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=1)

        predictions.extend(preds.cpu().numpy())
        true_labels.extend(batch['labels'].cpu().numpy())

print("\n📈 Результаты на тестовой выборке:")
print(classification_report(true_labels, predictions, digits=4))

print("✅ Обучение и тестирование завершены.")
