import pandas as pd
import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score

train_input_ids = np.load('trans_data/train_encodings_input_ids.npy')
train_attention_mask = np.load('trans_data/train_encodings_attention_mask.npy')

val_input_ids = np.load('trans_data/val_encodings_input_ids.npy')
val_attention_mask = np.load('trans_data/val_encodings_attention_mask.npy')

test_input_ids = np.load('trans_data/test_encodings_input_ids.npy')
test_attention_mask = np.load('trans_data/test_encodings_attention_mask.npy')

train_df = pd.read_csv('../../dataset/train_data.csv')
val_df = pd.read_csv('../../dataset/val_data.csv')
test_df = pd.read_csv('../../dataset/test_data.csv')

train_labels = train_df['label'].values
val_labels = val_df['label'].values
test_labels = test_df['label'].values


class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            'input_ids': torch.tensor(self.input_ids[idx]),
            'attention_mask': torch.tensor(self.attention_mask[idx]),
            'labels': torch.tensor(self.labels[idx])
        }
        return item


train_dataset = EmotionDataset(train_input_ids, train_attention_mask, train_labels)
val_dataset = EmotionDataset(val_input_ids, val_attention_mask, val_labels)
test_dataset = EmotionDataset(test_input_ids, test_attention_mask, test_labels)


def compute_metrics(p):
    preds = p.predictions.argmax(axis=-1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}


from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',  # директория для результатов
    num_train_epochs=3,  # эпохи
    per_device_train_batch_size=16,  # батч для обучения
    per_device_eval_batch_size=64,  # батч для валидации
    warmup_steps=500,  # шаги для разогрева
    weight_decay=0.01,  # регуляризация
    logging_dir='./logs',  # логи
    eval_strategy='epoch',  # оценка каждой эпохи
    save_strategy='epoch',  # сохранение каждой эпохи
    load_best_model_at_end=True,  # загружать лучшую модель в конце
    metric_for_best_model='f1',  # F1-метрика для оценки лучшей модели
)


# Инициализация модели
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=7)  # 7 классов

# Создание объекта Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Обучение модели
trainer.train()

# Оценка модели на тестовых данных
trainer.evaluate(test_dataset)

model.save_pretrained('./model_save')  # Путь для сохранения модели


print("Модель и токенизатор сохранены.")

'''from transformers import BertForSequenceClassification, BertTokenizer

# Загрузка сохраненной модели и токенизатора
model = BertForSequenceClassification.from_pretrained('./model_save')
tokenizer = BertTokenizer.from_pretrained('./model_save')

print("Модель и токенизатор загружены.")'''
