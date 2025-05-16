import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, DataCollatorWithPadding
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

# устройство
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Устройство:", device)

# датасет
df = pd.read_csv("../../dataset/updated_data_cleaned2.csv")  # clean_text, label

# раздел
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)

print(f"Сохранено: train.csv ({len(train_df)}), val.csv ({len(val_df)}), test.csv ({len(test_df)})")

# токенизатор + датасеты

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



# загрузка через dataloader
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_dataset = EmotionDataset(train_df['ru_text'], train_df['label'], tokenizer)
val_dataset = EmotionDataset(val_df['ru_text'], val_df['label'], tokenizer)
test_dataset = EmotionDataset(test_df['ru_text'], test_df['label'], tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=data_collator)
val_loader = DataLoader(val_dataset, batch_size=8, collate_fn=data_collator)
test_loader = DataLoader(test_dataset, batch_size=8, collate_fn=data_collator)


# настройки
model = BertForSequenceClassification.from_pretrained("DeepPavlov/rubert-base-cased", num_labels=7)
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

# обучение
epochs = 3

for epoch in range(epochs):
    print(f"\n!!!!!!Эпоха {epoch + 1}")
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / len(train_loader)
    print(f"Средний лосс: {avg_loss:.4f}")

    # валидация
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch['labels'].cpu().numpy())

    print("\nВалидация:")
    print(classification_report(true_labels, predictions, digits=4))

# финальное тестирование

model.eval()
predictions, true_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        predictions.extend(preds.cpu().numpy())
        true_labels.extend(batch['labels'].cpu().numpy())

print("\nТестовая выборка:")
print(classification_report(true_labels, predictions, digits=4))

model.save_pretrained("emotion_bert_model")
tokenizer.save_pretrained("emotion_bert_model")
print("Модель и токенизатор сохранены в папке emotion_bert_model")