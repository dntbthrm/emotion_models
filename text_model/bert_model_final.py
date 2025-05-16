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


# обучение на гпу
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("устройство:", device)
torch.cuda.empty_cache()
gc.collect()
# датасет
df = pd.read_csv("../../dataset/updated_data_cleaned2.csv")  # clean_text, label

train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

train_df.to_csv("train1.csv", index=False)
val_df.to_csv("val1.csv", index=False)
test_df.to_csv("test1.csv", index=False)

print(f"Сохранено: train1.csv ({len(train_df)}), val1.csv ({len(val_df)}), test1.csv ({len(test_df)})")

# токенизация + датасеты
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

# веса
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df['label']),
    y=train_df['label']
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print(f"веса классов: {class_weights}")

# настройки
model = BertForSequenceClassification.from_pretrained("DeepPavlov/rubert-base-cased", num_labels=7)
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = CrossEntropyLoss(weight=class_weights)

from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, verbose=True)

# обучение
epochs = 10
best_val_f1 = 0
early_stop_counter = 0
early_stop_patience = 3

for epoch in range(epochs):
    print(f"\n!!!!Эпоха {epoch + 1}/{epochs}")
    model.train()
    total_loss = 0

    for step, batch in enumerate(tqdm(train_loader, desc="---Обучение")):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = loss_fn(outputs.logits, batch['labels'])
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (step + 1) % 200 == 0:
            print(f"  -> [{step+1}/{len(train_loader)}] Лосс: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"Средний лосс за эпоху: {avg_loss:.4f}")

    # === 6. Валидация ===
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="@Валидация"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch['labels'].cpu().numpy())

    val_f1 = f1_score(true_labels, predictions, average='weighted')
    print(f"\nVal F1-score: {val_f1:.4f}")
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

# финальное тестирование
print("\n!!!!!!!Финальное тестирование на test.csv!!!!!!!!!!")
model = BertForSequenceClassification.from_pretrained("emotion_bert_best").to(device)
model.eval()
predictions, true_labels = [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Тест"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=1)

        predictions.extend(preds.cpu().numpy())
        true_labels.extend(batch['labels'].cpu().numpy())

print("\nРезультаты на тестовой выборке:")
print(classification_report(true_labels, predictions, digits=4))

print("Обучение и тестирование завершены.")
