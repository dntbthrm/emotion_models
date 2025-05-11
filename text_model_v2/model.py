import pandas as pd
import torch
from sklearn.utils import shuffle
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import os
os.environ["WANDB_DISABLED"] = "true"

# ✅ Загрузка и балансировка датасета
df = pd.read_csv('../../dataset/updated_data_cleaned.csv')  # ← Заменить на имя твоего CSV-файла

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
    return tokenizer(batch['ru_text'], padding='longest', truncation=True, max_length=128)

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

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
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
print(f"📊 Accuracy: {metrics['eval_accuracy']:.4f}, F1-macro: {metrics['eval_f1_macro']:.4f}")